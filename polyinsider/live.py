"""Read-only Gamma+Data API collector, time-frozen predictions, audit chain, evaluation.

This module does not send orders, trade, or assert the presence of private info.
"""
from __future__ import annotations
import csv, json, hashlib, os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import requests
from .core import market_features, model_probability, wallet_signals, metrics
from .wallet_model import prob_wallet

GAMMA='https://gamma-api.polymarket.com'
DATA='https://data-api.polymarket.com'
LOG_FIELDS=['captured_at','event_id','condition_id','question','market_yes_price','predicted_yes',
            'model_source','mode','model_sha256','wallet_model_source','observed_fill_count']

def now_iso():return datetime.now(timezone.utc).isoformat()

def as_list(x):
    if isinstance(x,list):return x
    if not x:return []
    try:return json.loads(x)
    except (ValueError,TypeError):return []

def get_json(session,url,params=None):
    r=session.get(url,params=params,timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_open(session=None,limit=8):
    s=session or requests.Session()
    markets=get_json(s,GAMMA+'/markets',{'active':'true','closed':'false',
        'order':'volumeNum','ascending':'false','limit':min(80,max(12,limit*3))})
    if not isinstance(markets,list):raise ValueError('Unexpected Gamma markets schema')
    ans=[]
    for m in markets:
        if m.get('closed') is not False or m.get('active') is not True:continue
        outcomes=as_list(m.get('outcomes'));prices=as_list(m.get('outcomePrices'))
        if len(outcomes)!=2 or len(prices)!=2:continue
        idx=next((i for i,x in enumerate(outcomes) if str(x).lower()=='yes'),None)
        if idx is None:continue
        try:p=float(prices[idx])
        except (TypeError,ValueError,IndexError):continue
        if not .001<p<.999:continue
        evs=m.get('events') or []
        ev=str(evs[0]['id']) if evs and isinstance(evs[0],dict) and 'id' in evs[0] else str(m.get('id'))
        ans.append({'event_id':ev,'condition_id':str(m.get('conditionId') or ''),
             'question':m.get('question',''),'yes_price':p,
             'volume_usd':float(m.get('volumeNum') or m.get('volume') or 0),
             'end_date':m.get('endDate'),'slug':m.get('slug',''),
             'token_ids':as_list(m.get('clobTokenIds'))})
        if len(ans)>=limit:break
    return ans

def fetch_taker_trades(session,condition,limit=500,lookback_hours=48,max_pages=4,asof=None):
    """Return observations strictly before capture from most recent v2 taker fill pages.

    V2 is keyset-paginated. Do not assume a single page captures all wallet flow.
    This remains a recent-flow proxy, not a full order book or inventory.
    """
    asof=asof or datetime.now(timezone.utc)
    boundary=int((asof-timedelta(hours=lookback_hours)).timestamp())
    cap=int(asof.timestamp())
    ans=[];cursor=None
    for _ in range(max_pages):
        params={'condition':condition,'limit':limit,'taker_only':'true'}
        if cursor:params={'cursor':cursor}
        obj=get_json(session,DATA+'/v2/trades',params)
        if not isinstance(obj,dict) or not isinstance(obj.get('data'),list):
            raise ValueError('Unexpected v2 trades schema; refusing to create a prediction')
        records=obj['data']
        oldest=None
        for t in records:
            wallet=t.get('proxy_wallet') or t.get('proxyWallet')
            side=str(t.get('side','')).upper();outcome=str(t.get('outcome','')).lower()
            if outcome not in ['yes','no'] or side not in ['BUY','SELL'] or not wallet:continue
            try:
                stamp=int(t['timestamp']);p=float(t['price']);amount=float(t['size'])*p
            except (ValueError,KeyError,TypeError):continue
            oldest=stamp if oldest is None else min(oldest,stamp)
            if stamp>cap or stamp<boundary or not 0<p<1 or amount<=0:continue
            yes_direction=(outcome=='yes') == (side=='BUY')
            ans.append({'wallet':str(wallet).lower(),
                   'direction':'YES' if yes_direction else 'NO','amount_usd':amount,'timestamp':stamp})
        pagination=obj.get('pagination') or {}
        cursor=pagination.get('next_cursor') or obj.get('next_cursor')
        if not cursor or not records or (oldest is not None and oldest<boundary):break
    return ans

def _wallet_map(wallet_probability_records):
    if wallet_probability_records is None:return {}
    if isinstance(wallet_probability_records,dict):return wallet_probability_records
    return {r['wallet']:r for r in wallet_probability_records}

def score_market(market,trades,model,wallet_stats,captured_at=None,
                 mode='LIVE PUBLIC API',wallet_probability_records=None,model_sha256=''):
    if market.get('closed') is True:raise ValueError('Cannot forecast a closed market')
    time=captured_at or now_iso()
    state={k:tuple(v) for k,v in wallet_stats.items()}
    f=market_features(market['yes_price'],trades,state)
    p=model_probability(model,f)
    scored=wallet_signals(trades,state)
    wp=_wallet_map(wallet_probability_records)
    for item in scored:
        meta=wp.get(item['wallet'])
        if meta is not None:
            item['prob_future_outperform']=round(float(meta['prob_future_outperform']),4)
            # Previously reported historical flags are separate from probability estimates.
            item['probability_label']='forecast of future above-price advantage'
        else:item['prob_future_outperform']=None
    return {'captured_at':time,'event_id':market['event_id'],'condition_id':market['condition_id'],
            'question':market['question'],'slug':market.get('slug',''),'end_date':market.get('end_date'),
            'volume_usd':market.get('volume_usd',0),'market_yes_price':market['yes_price'],
            'predicted_yes':round(p,4),'prediction_vs_market_pp':round((p-market['yes_price'])*100,2),
            'n_taker_trade_rows':len(trades),'features':f,
            'top_wallets':scored[:12],
            'model_source':model['source'],'model_sha256':model_sha256,'mode':mode,
            'wallet_model_source': 'none' if not wp else next(iter(wp.values())).get('source','trained wallet model'),
            'caveat':'Recent public taker fills only; probabilistic future skill, never proof of private information.'}

def append_predictions(preds,csv_path):
    path=Path(csv_path);path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as fd:
        writer=csv.DictWriter(fd,fieldnames=LOG_FIELDS)
        if not exists:writer.writeheader()
        for p in preds:
            writer.writerow({k:p.get(k,'') for k in LOG_FIELDS})
        fd.flush();os.fsync(fd.fileno())

def append_audit_chain(preds,path):
    """Append-only convention with simple SHA-256 hash chain (not tamper-proof signing)."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    previous='GENESIS'
    if path.exists() and path.stat().st_size:
        with path.open(encoding='utf8') as f:
            for line in f:
                if line.strip():previous=json.loads(line)['entry_sha256']
    with path.open('a',encoding='utf8') as f:
        for p in preds:
            content=json.dumps(p,sort_keys=True,separators=(',',':'))
            digest=hashlib.sha256((previous+'|'+content).encode()).hexdigest()
            record={'previous_sha256':previous,'entry_sha256':digest,'prediction':p}
            f.write(json.dumps(record,sort_keys=True)+'\n');previous=digest
        f.flush();os.fsync(f.fileno())

def verify_audit_chain(path):
    previous='GENESIS';total=0
    with Path(path).open() as f:
        for i,line in enumerate(f,1):
            r=json.loads(line)
            content=json.dumps(r['prediction'],sort_keys=True,separators=(',',':'))
            expected=hashlib.sha256((previous+'|'+content).encode()).hexdigest()
            if r['previous_sha256']!=previous or r['entry_sha256']!=expected:
                raise ValueError(f'Prediction audit chain corrupted at row {i}')
            previous=expected;total+=1
    return {'verified_rows':total,'last_sha256':previous,'note':'Hash-chain integrity only; no trusted external time-stamp or signature.'}

def capture_live(model_path,web_dir,limit=8,session=None,allow_demo_model=False,
                 wallet_model_path=None):
    web=Path(web_dir);web.mkdir(parents=True,exist_ok=True)
    base=Path(model_path);model=json.loads(base.read_text())
    synthetic=model['source'].upper().startswith('SYNTHETIC')
    if synthetic and not allow_demo_model:
        raise ValueError('Synthetic model cannot be presented as real forecasts; train on real history or explicitly allow demo.')
    state=json.loads((base.parent/'wallet_state.json').read_text())
    wallet_data={}; wallet_source='NOT TRAINED'
    wallet_model_path=Path(wallet_model_path) if wallet_model_path else base.parent/'wallet_model.json'
    if wallet_model_path.exists():
        wallet_model=json.loads(wallet_model_path.read_text());wallet_source=wallet_model.get('source','UNSPECIFIED')
        wallet_file=wallet_model_path.parent/'wallet_probabilities.csv'
        if wallet_file.exists():
            import pandas as pd
            wallet_data={r['wallet']:{**r,'source':wallet_source} for r in pd.read_csv(wallet_file).to_dict('records')}
    if synthetic and wallet_data and not wallet_source.upper().startswith('SYNTHETIC'):
        raise ValueError('Refusing to pair synthetic market model with mismatched wallet model in demo mode')
    s=session or requests.Session()
    markets=fetch_open(s,limit)
    preds=[];failures=[];stamp=now_iso();capture_at=datetime.fromisoformat(stamp)
    digest=hashlib.sha256(base.read_bytes()).hexdigest()
    for m in markets:
        if not m['condition_id']:continue
        try:
            trades=fetch_taker_trades(s,m['condition_id'],asof=capture_at)
            # Must have at least a few genuinely observed earlier fills to score a market.
            if len(trades)<3:raise ValueError('Fewer than 3 verified pre-snapshot fill rows')
            mode='PUBLIC MARKET DATA + SYNTHETIC MODEL — TEST ONLY' if synthetic else 'LIVE PUBLIC API — HISTORICAL MODEL'
            preds.append(score_market(m,trades,model,state,captured_at=stamp,mode=mode,
                          wallet_probability_records=wallet_data,model_sha256=digest))
        except Exception as exc:failures.append({'condition_id':m['condition_id'],'error':str(exc)[:220]})
    if not preds:
        raise RuntimeError(f'No defensible predictions captured; first errors: {failures[:3]}')
    status={'updated_at':stamp,'mode': 'LIVE API + SYNTHETIC MODEL — NOT VALIDATED' if synthetic else 'LIVE PUBLIC API — REAL MODEL',
       'model_source':model['source'],'wallet_model_source':wallet_source,
       'markets':preds,'errors':failures,
       'note':'Read-only real public API poll. Wallet/model training provenance shown separately. Each snapshot logged before resolution.'}
    (web/'data').mkdir(exist_ok=True)
    dest=web/'data'/'live.json';tmp=dest.with_suffix('.json.tmp')
    # Log first; atomic site refresh happens only after durable persistence.
    append_predictions(preds,base.parent/'predictions.csv')
    append_audit_chain(preds,base.parent/'predictions_audit.jsonl')
    tmp.write_text(json.dumps(status,indent=2),encoding='utf-8');os.replace(tmp,dest)
    return status

def evaluate_saved_predictions(log_path,resolutions_csv,output_path):
    import pandas as pd
    p=pd.read_csv(log_path);r=pd.read_csv(resolutions_csv)
    need=['condition_id','resolved_yes','resolution_time']
    if not set(need)<=set(r.columns):raise ValueError(f'Resolution file needs: {need}')
    if r.condition_id.duplicated().any():raise ValueError('Duplicate resolution ID')
    if not set(['captured_at','condition_id','predicted_yes','market_yes_price'])<=set(p.columns):
        raise ValueError('Prediction file lacks required fields')
    j=p.merge(r[need],on='condition_id',how='inner')
    j['captured_at']=pd.to_datetime(j.captured_at,utc=True)
    j['resolution_time']=pd.to_datetime(j.resolution_time,utc=True)
    j=j[(j.captured_at<j.resolution_time)&j.resolved_yes.isin([0,1])]
    if j.empty:raise ValueError('No verified forecasts captured before actual resolution')
    j=j.sort_values('captured_at').drop_duplicates('condition_id',keep='first')
    result={'n_verified_pre_resolution_markets':len(j),
       'model':metrics(j.resolved_yes,j.predicted_yes),
       'market_baseline':metrics(j.resolved_yes,j.market_yes_price),
       'all_predictions_pre_resolution':True,
       'note':'First saved forecast per condition; timestamp must predate independently verified actual resolution.'}
    out=Path(output_path);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2));return result
