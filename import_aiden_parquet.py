#!/usr/bin/env python3
"""Adapt Aiden's ECE473 event Parquet data for PolyInsiders' 10-column pre-resolution CSV.

READ-ONLY with respect to the source folder. Outputs CSV and a provenance report.
This constructs an executed-fill-flow proxy; it is not a settled wallet PnL ledger.
History is sourced from Aiden's metadata, not independently verified resolution timestamps.
"""
import argparse
import csv
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def utc(series):
    return pd.to_datetime(series, utc=True, errors='coerce')


def main():
    p=argparse.ArgumentParser(description='Convert Aiden 2026 data_events/live Parquet to PolyInsiders history')
    p.add_argument('--input', default='/Users/emilly/Desktop/pm473/data_events/live')
    p.add_argument('--out', default='data/prepared_history.csv')
    p.add_argument('--lead-hours', type=int, default=24,
                   help='Freeze at least this many hours before recorded information release / resolution')
    p.add_argument('--lookback-hours', type=int, default=48)
    p.add_argument('--max-markets', type=int, default=0, help='Optional diagnostic subset; 0=all')
    p.add_argument('--dry-run', action='store_true', help='Only audit metadata; do not scan historical fills')
    args=p.parse_args()
    src=Path(args.input)
    for n in ('fills_clean.parquet', 'markets_coded.parquet', 'wallet_codes.parquet'):
        if not (src/n).is_file():p.error(f'Missing {src/n}')
    out=Path(args.out)
    if src.resolve() in out.resolve().parents:
        p.error('Choose an output outside of Aiden\'s source folder; source data must remain untouched')
    if args.lead_hours<=0 or args.lookback_hours<=0:p.error('lead/lookback must be positive')

    m=pd.read_parquet(src/'markets_coded.parquet')
    required={'condition_id','condition_id_str','outcome_0','outcome_1','winner_index',
              'resolve_time','event_id','question','start_time','info_time'}
    if miss:=required-set(m.columns):p.error(f'Metadata missing: {sorted(miss)}')
    m['resolve_time']=utc(m['resolve_time']);m['info_time']=utc(m['info_time']);m['start_time']=utc(m['start_time'])
    # Skip nonbinary markets and those with a non-YES/NO outcome convention.
    m['o0']=m.outcome_0.astype(str).str.strip().str.lower()
    m['o1']=m.outcome_1.astype(str).str.strip().str.lower()
    yes_is_0=(m.o0.eq('yes')&m.o1.eq('no'))
    yes_is_1=(m.o1.eq('yes')&m.o0.eq('no'))
    m=m.loc[(yes_is_0|yes_is_1)&m.winner_index.isin([0,1])&m.resolve_time.notna()].copy()
    m['yes_index']=np.where(m.o0.eq('yes'),0,1)
    m['resolved_yes']=(m.winner_index.astype(int)==m.yes_index).astype(int)
    # Checkpoint is BEFORE the earliest recorded information time and market resolution.
    m['reference_time']=m[['info_time','resolve_time']].min(axis=1)
    m['snapshot_time']=m.reference_time-pd.Timedelta(hours=args.lead_hours)
    # Require a complete 48-hour lookback during the market's declared life.
    m=m.loc[(m.snapshot_time>=m.start_time+pd.Timedelta(hours=args.lookback_hours))&
            (m.snapshot_time<m.resolve_time)&m.event_id.notna()&m.condition_id_str.notna()].copy()
    if args.max_markets:
        m=m.sort_values('resolve_time').head(args.max_markets).copy()
    if m.empty:p.error('No eligible binary YES/NO markets with a valid pre-information 48h window')
    if m.condition_id.duplicated().any():p.error('Duplicate coded market IDs')
    if m.condition_id_str.duplicated().any():p.error('Duplicate original condition IDs')

    fields=['condition_id','condition_id_str','event_id','question','snapshot_time',
            'resolve_time','yes_index','resolved_yes']
    snap=m[fields].set_index('condition_id')
    print('Eligible binary YES/NO markets:', len(snap), flush=True)
    print('Resolved YES / NO:', m.resolved_yes.value_counts().sort_index().to_dict(), flush=True)
    print('First/last snapshots:', str(m.snapshot_time.min()), str(m.snapshot_time.max()), flush=True)
    print('Metadata information-time source:',m.get('info_time_source',pd.Series(dtype=str)).value_counts(dropna=False).head(8).to_dict(),flush=True)
    if args.dry_run:
        print('DRY RUN: no files written; inspect metadata before proceeding.')
        return

    snap_ns={int(k):int(v.snapshot_time.value) for k,v in snap.iterrows()}
    yes_idx={int(k):int(v.yes_index) for k,v in snap.iterrows()}
    # Joining via small dictionaries avoids large full-dataset merges.
    pf=pq.ParquetFile(src/'fills_clean.parquet')
    cols=set(pf.schema_arrow.names)
    need={'wallet','condition_id','price','size','ts','outcome_index','is_buy'}
    if miss:=need-cols:p.error(f'Fills missing {sorted(miss)}')
    extra=['dir0'] if 'dir0' in cols else []
    out.parent.mkdir(parents=True,exist_ok=True)
    db=out.with_suffix('.work.sqlite')
    if db.exists():p.error(f'Temporary database already exists: {db}; move it out of the way after inspection')
    con=sqlite3.connect(db)
    con.execute('PRAGMA journal_mode = MEMORY')
    con.execute('CREATE TABLE flows (condition_id INTEGER, wallet INTEGER, usd REAL, PRIMARY KEY(condition_id,wallet))')
    con.execute('CREATE TABLE prices (condition_id INTEGER PRIMARY KEY, timestamp_ns INTEGER, yes_price REAL)')
    total=0;recent=0;eligible=0;dir_matches=0;dir_checked=0
    try:
        for i,rb in enumerate(pf.iter_batches(batch_size=150_000,columns=sorted(need)+extra),1):
            d=rb.to_pandas()
            total+=len(d)
            d=d[d.condition_id.isin(snap_ns)].copy()
            if d.empty:continue
            eligible+=len(d)
            tt=utc(d.ts).astype('int64')
            freeze=d.condition_id.map(snap_ns).astype('int64')
            mask=((tt<=freeze)&(tt>=freeze-args.lookback_hours*3600*10**9)&
                  (d.price>0)&(d.price<1)&(d['size']>0)&d.wallet.notna()&
                  d.outcome_index.isin([0,1]))
            d=d.loc[mask].copy()
            if d.empty:continue
            recent+=len(d)
            d['when_ns']=tt[mask].to_numpy(dtype=np.int64)
            d['yes_index']=d.condition_id.map(yes_idx).astype(int)
            buy_yes=((d.outcome_index.to_numpy()==d.yes_index.to_numpy())==d.is_buy.to_numpy())
            d['flow']=np.where(buy_yes,1.,-1.)*d['size'].to_numpy()*d.price.to_numpy()
            d['yes_price']=np.where(d.outcome_index.to_numpy()==d.yes_index.to_numpy(), d.price.to_numpy(),1-d.price.to_numpy())
            if 'dir0' in d.columns:
                # Aiden's dir0 should have the same sign if it indicates net token_0 demand.
                token0_buy=((d.outcome_index.to_numpy()==0)==d.is_buy.to_numpy())
                valid=np.isin(d.dir0.to_numpy(),[-1,1])
                dir_matches+=int((d.dir0.to_numpy()[valid]==np.where(token0_buy[valid],1,-1)).sum())
                dir_checked+=int(valid.sum())
            grouped=d.groupby(['condition_id','wallet'],sort=False)['flow'].sum()
            con.executemany('INSERT INTO flows VALUES(?,?,?) ON CONFLICT(condition_id,wallet) DO UPDATE SET usd=usd+excluded.usd',
                            [(int(c),int(w),float(f)) for (c,w),f in grouped.items()])
            latest=d.sort_values('when_ns').groupby('condition_id',sort=False).tail(1)
            con.executemany('INSERT INTO prices VALUES(?,?,?) ON CONFLICT(condition_id) DO UPDATE SET timestamp_ns=excluded.timestamp_ns,yes_price=excluded.yes_price WHERE excluded.timestamp_ns > prices.timestamp_ns',
                            [(int(x.condition_id),int(x.when_ns),float(x.yes_price)) for x in latest.itertuples()])
            con.commit()
            if i%12==0:print(f'Scanned {total:,} rows; {recent:,} eligible pre-freeze fills',flush=True)
        if dir_checked:
            match=dir_matches/dir_checked
            print(f'dir0 consistency with token_0 demand: {match:.1%}',flush=True)
            if match<.95:
                raise RuntimeError('Ambiguous is_buy/outcome_index semantics. Stopping rather than mislabeling the flow.')

        codes=pd.read_parquet(src/'wallet_codes.parquet',columns=['code','wallet'])
        wallets=dict(zip(codes.code.astype(int),codes.wallet.astype(str).str.lower()))
        if len(wallets)!=len(codes):raise RuntimeError('Duplicate wallet codes')
        prices=dict(con.execute('SELECT condition_id,yes_price FROM prices'))
        count=0;markets=set();wallet_ids=set();missing_wallet=0
        with out.open('w',newline='',encoding='utf-8') as fh:
            writer=csv.writer(fh)
            writer.writerow(['event_id','condition_id','title','snapshot_time','resolution_time',
                             'yes_price','wallet','direction','amount_usd','resolved_yes'])
            for c,w,f in con.execute('SELECT condition_id,wallet,usd FROM flows WHERE ABS(usd)>=0.01'):
                r=snap.loc[c];p0=prices.get(c)
                wallet=wallets.get(w)
                if wallet is None:missing_wallet+=1;continue
                if p0 is None or not .001<p0<.999:continue
                writer.writerow([str(r.event_id),str(r.condition_id_str),str(r.question),
                                 r.snapshot_time.isoformat(),r.resolve_time.isoformat(),
                                 round(p0,6),wallet,'YES' if f>0 else 'NO',round(abs(f),5),int(r.resolved_yes)])
                count+=1;markets.add(c);wallet_ids.add(w)
        report={'source':'Aiden pm473 data_events/live, local Parquet archive',
                'source_folder':str(src),'input_markets_eligible_metadata':len(m),
                'input_fills_scanned':total,'fills_from_selected_markets':eligible,
                'fills_in_historical_lookback':recent,'output_markets':len(markets),
                'output_wallet_market_rows':count,'output_unique_wallets':len(wallet_ids),
                'missing_wallet_code_rows':missing_wallet,'snapshot_lead_hours':args.lead_hours,
                'snapshot_lookback_hours':args.lookback_hours,
                'event_time_sources':m.get('info_time_source',pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
                'output_csv':str(out),
                'limitations':['This is actual archived fill flow, not wallet balance, private-information labels, or audited wallet PnL.',
                    'Resolution times and winning outcomes are Aiden\'s archived metadata, not independently verified here.',
                    'Information timing is often a scheduled-end proxy; a pre-resolution snapshot may still follow earlier news.',
                    'Last fill price is a historical trade proxy, not a historical order-book midpoint.',
                    'Only earlier fills enter feature construction, but outcome labels are used later in supervised training.',
                    'Wallet event-level score from executed flow is distinct from Aiden\'s audited wallet_market_pnl.parquet.']}
        report_file=out.with_suffix('.import_report.json')
        report_file.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        if not count:raise RuntimeError('No market-wallet rows written; adjust eligibility or check fill semantics')
    finally:
        con.close()
        if db.exists():db.unlink()

if __name__=='__main__':main()
