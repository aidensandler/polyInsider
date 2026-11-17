"""Convert public executed-fill CSV + verified market metadata to frozen pre-resolution model snapshots.
This does NOT reconstruct wallet positions or a historical order book. Trades-only proxy baseline.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

NEED_TRADES={'condition_id','proxy_wallet','outcome','side','price','size','timestamp'}
NEED_MARKETS={'condition_id','event_id','title','snapshot_time','resolution_time','resolved_yes'}

def parse_time(s):
    nums=pd.to_numeric(s,errors='coerce')
    if nums.notna().all():return pd.to_datetime(nums,unit='s',utc=True)
    return pd.to_datetime(s,utc=True,errors='raise')

def prepare_snapshots(trades_csv,metadata_csv,out_csv,lookback_hours=48,price_max_age_hours=24):
    t=pd.read_csv(trades_csv); m=pd.read_csv(metadata_csv)
    t=t.rename(columns={'proxyWallet':'proxy_wallet','conditionId':'condition_id'})
    if missing:=NEED_TRADES-set(t.columns):raise ValueError(f'Trades missing {sorted(missing)}')
    if missing:=NEED_MARKETS-set(m.columns):raise ValueError(f'Metadata missing {sorted(missing)}')
    m['snapshot_time']=parse_time(m.snapshot_time);m['resolution_time']=parse_time(m.resolution_time)
    t['timestamp']=parse_time(t.timestamp)
    if (m.snapshot_time>=m.resolution_time).any():raise ValueError('Snapshots must predate actual verified resolution')
    if m.condition_id.duplicated().any():raise ValueError('Market metadata must be unique by condition_id')
    if not m.resolved_yes.isin([0,1]).all():raise ValueError('Use verified 0/1 settlement labels only')
    t['price']=pd.to_numeric(t.price,errors='coerce')
    t['size']=pd.to_numeric(t['size'],errors='coerce')
    t['side']=t.side.astype(str).str.upper()
    t['outcome']=t.outcome.astype(str).str.lower()
    t=t[(t.side.isin(['BUY','SELL']))&(t.outcome.isin(['yes','no']))&
       (t.price.between(.001,.999))&(t['size']>0)]
    # Some data stores maker and taker rows for the same fill; caller must deduplicate according to provenance.
    grouped={cond:g for cond,g in t.groupby('condition_id')}
    records=[];skips=[]
    for _,market in m.iterrows():
        g=grouped.get(market.condition_id)
        if g is None:skips.append((market.condition_id,'no fills'));continue
        freeze=market.snapshot_time
        g=g[(g.timestamp<=freeze)&(g.timestamp>=freeze-pd.Timedelta(hours=lookback_hours))].copy()
        if g.empty:skips.append((market.condition_id,'no fills before frozen snapshot'));continue
        # Public price proxy: most recent executed YES fill; if none, complement last NO fill.
        forprice=g[g.timestamp>=freeze-pd.Timedelta(hours=price_max_age_hours)].sort_values('timestamp')
        if forprice.empty:skips.append((market.condition_id,'no contemporaneous fill'));continue
        last=forprice.iloc[-1]
        p=float(last.price if last.outcome=='yes' else 1-last.price)
        if not .001<p<.999:skips.append((market.condition_id,'invalid price proxy'));continue
        g['signed_exposure']=g['size']*g.price*np.where(
             ((g.outcome=='yes')&(g.side=='BUY'))|((g.outcome=='no')&(g.side=='SELL')),1,-1)
        flows=g.groupby('proxy_wallet').signed_exposure.sum()
        for wallet,flow in flows.items():
            if abs(flow)<.01:continue
            records.append({'event_id':market.event_id,'condition_id':market.condition_id,
               'title':market.title,'snapshot_time':freeze.isoformat(),
               'resolution_time':market.resolution_time.isoformat(),
               'yes_price':round(p,6),'wallet':str(wallet).lower(),
               'direction':'YES' if flow>0 else 'NO','amount_usd':round(abs(float(flow)),4),
               'resolved_yes':int(market.resolved_yes)})
    if not records:raise ValueError('No valid pre-resolution examples; check timestamp coverage')
    out=Path(out_csv);out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out,index=False)
    report={'input_trades':int(len(t)),'input_markets':int(len(m)),
       'produced_markets':len({r['condition_id'] for r in records}),
       'produced_wallet_market_rows':len(records),
       'skipped_markets':len(skips),'first_skips':skips[:20],
       'price_proxy':'most recent fill near snapshot; not historical order-book midpoint',
       'flow_proxy':'signed executed USDC in historical lookback; not inventory or round-trip ROI',
       'warning':'Never interpret this dataset as the original wallet-level PnL / settlement ledger.'}
    out.with_suffix('.prep_report.json').write_text(json.dumps(report,indent=2))
    return report

def prepare_snapshots_streaming(trades_csv,metadata_csv,out_csv,lookback_hours=48,
                                 price_max_age_hours=24,chunksize=150_000):
    """Large-archive version of `prepare_snapshots`, using bounded-memory CSV chunks + SQLite.

    Can handle millions of taker fills without loading all of them into pandas.
    Each chunk is filtered to the exact historical frozen lookback first, then
    grouped into wallet/market net YES exposure on disk. Last trade price is a proxy,
    NOT historical top-of-book. Does not deduplicate legitimate identical-looking fills.
    """
    import sqlite3
    import os
    m=pd.read_csv(metadata_csv,dtype={'condition_id':str})
    if missing:=NEED_MARKETS-set(m.columns):raise ValueError(f'Metadata missing {sorted(missing)}')
    m['snapshot_time']=parse_time(m.snapshot_time);m['resolution_time']=parse_time(m.resolution_time)
    if m.condition_id.duplicated().any():raise ValueError('Duplicate market metadata ID')
    if not (m.snapshot_time<m.resolution_time).all():raise ValueError('Freeze time must precede actual resolution')
    if not m.resolved_yes.isin([0,1]).all():raise ValueError('Resolution labels must be verified binary 0/1')
    root=Path(out_csv);root.parent.mkdir(parents=True,exist_ok=True)
    db=root.with_suffix('.work.sqlite')
    if db.exists():db.unlink()
    con=sqlite3.connect(db)
    con.execute('PRAGMA journal_mode=OFF')
    con.execute('CREATE TABLE flows (condition_id TEXT, wallet TEXT, flow REAL, PRIMARY KEY(condition_id,wallet))')
    con.execute('CREATE TABLE prices (condition_id TEXT PRIMARY KEY, ts TEXT, price REAL)')
    metadata=m[['condition_id','snapshot_time']].copy()
    total=0;kept=0
    try:
        reader=pd.read_csv(trades_csv,chunksize=chunksize,compression='infer',low_memory=False)
        for batch in reader:
            total+=len(batch)
            batch=batch.rename(columns={'proxyWallet':'proxy_wallet','conditionId':'condition_id'})
            if missing:=NEED_TRADES-set(batch.columns):raise ValueError(f'Trades missing {sorted(missing)}')
            batch=batch[list(NEED_TRADES)].copy()
            batch['condition_id']=batch['condition_id'].astype(str)
            batch=batch.merge(metadata,on='condition_id',how='inner')
            if batch.empty:continue
            batch['timestamp']=parse_time(batch.timestamp)
            batch['price']=pd.to_numeric(batch.price,errors='coerce')
            batch['size']=pd.to_numeric(batch['size'],errors='coerce')
            batch['side']=batch.side.astype(str).str.upper()
            batch['outcome']=batch.outcome.astype(str).str.lower()
            batch=batch[(batch.side.isin(['BUY','SELL']))&(batch.outcome.isin(['yes','no']))&
              batch.price.between(.001,.999)&(batch['size']>0)&
              (batch.timestamp<=batch.snapshot_time)&
              (batch.timestamp>=batch.snapshot_time-pd.Timedelta(hours=lookback_hours))]
            if batch.empty:continue
            kept+=len(batch)
            # Signed YES / NO taker buying flow is not a wallet's complete position.
            yes_positive=((batch.outcome=='yes')&(batch.side=='BUY'))|((batch.outcome=='no')&(batch.side=='SELL'))
            batch['flow']=batch['size']*batch.price*np.where(yes_positive,1.,-1.)
            flows=batch.groupby(['condition_id','proxy_wallet']).flow.sum()
            con.executemany('INSERT INTO flows(condition_id,wallet,flow) VALUES(?,?,?) '
                'ON CONFLICT(condition_id,wallet) DO UPDATE SET flow=flow+excluded.flow',
                [(str(cid),str(wallet).lower(),float(value)) for (cid,wallet),value in flows.items()])
            contemporaneous=batch[batch.timestamp>=batch.snapshot_time-pd.Timedelta(hours=price_max_age_hours)]
            if not contemporaneous.empty:
                latest=contemporaneous.sort_values('timestamp').groupby('condition_id').tail(1)
                for r in latest.itertuples():
                    p=float(r.price if r.outcome=='yes' else 1-r.price)
                    con.execute('INSERT INTO prices(condition_id,ts,price) VALUES(?,?,?) '
                        'ON CONFLICT(condition_id) DO UPDATE SET ts=excluded.ts,price=excluded.price '
                        'WHERE excluded.ts>=prices.ts',(r.condition_id,r.timestamp.isoformat(),p))
            con.commit()
        meta_by_id=m.set_index('condition_id')
        produced=set();rows=0
        with root.open('w',newline='',encoding='utf8') as fd:
            columns=['event_id','condition_id','title','snapshot_time','resolution_time','yes_price',
                     'wallet','direction','amount_usd','resolved_yes']
            writer=__import__('csv').DictWriter(fd,fieldnames=columns);writer.writeheader()
            query='SELECT f.condition_id,f.wallet,f.flow,p.price FROM flows f JOIN prices p USING(condition_id) WHERE ABS(f.flow)>=0.01'
            for cid,wallet,flow,price in con.execute(query):
                if not 0.001<price<.999:continue
                market=meta_by_id.loc[cid]
                writer.writerow({'event_id':market.event_id,'condition_id':cid,'title':market.title,
                    'snapshot_time':market.snapshot_time.isoformat(),
                    'resolution_time':market.resolution_time.isoformat(),
                    'yes_price':round(price,6),'wallet':wallet,
                    'direction':'YES' if flow>0 else 'NO','amount_usd':round(abs(flow),4),
                    'resolved_yes':int(market.resolved_yes)})
                produced.add(cid);rows+=1
        if not rows:raise ValueError('No eligible historical wallet snapshots; check market IDs and lookback')
        report={'source':'HISTORICAL EXECUTED FILLS — STREAMING IMPORT',
             'input_fill_rows':total,'pre_snapshot_fill_rows_in_lookback':kept,
             'input_markets':len(m),'produced_markets':len(produced),
             'produced_wallet_market_rows':rows,'skipped_markets':len(m)-len(produced),
             'limitations':'Last actual fill is a price proxy; signed recent flow is not position PnL. Verify provenance and complete event grouping.'}
        root.with_suffix('.prep_report.json').write_text(json.dumps(report,indent=2))
        return report
    finally:
        con.close()
        if db.exists():os.unlink(db)
