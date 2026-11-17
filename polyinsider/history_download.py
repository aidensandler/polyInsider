"""Retrieve historical public executed taker fills for independently verified market list.

Requires external network on user's computer. Full v2 cursor pagination; no guessed
resolutions or end times. This is not itself a trained empirical model.
"""
import csv, json, time
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import requests
from .live import DATA

FIELDS=['condition_id','proxy_wallet','outcome','side','price','size','timestamp','transaction_hash']

def download_v2_fills(markets_csv,out_csv,max_pages_per_market=300,delay=0.10,session=None):
    markets=pd.read_csv(markets_csv,dtype={'condition_id':str})
    if 'condition_id' not in markets.columns:raise ValueError('Market metadata must supply condition_id')
    if markets.condition_id.isna().any() or markets.condition_id.duplicated().any():
        raise ValueError('Market identifiers missing or duplicated')
    if max_pages_per_market<=0:raise ValueError('max_pages_per_market must be positive')
    s=session or requests.Session();dest=Path(out_csv);dest.parent.mkdir(parents=True,exist_ok=True)
    failures=[];total=0;incomplete=[];condition_page_counts={}
    with dest.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for market_id in markets.condition_id.astype(str):
            cursor=None;pages=0;seen=set()
            while pages<max_pages_per_market:
                params={'condition':market_id,'limit':500,'taker_only':'true'}
                if cursor:params={'cursor':cursor}
                try:
                    response=s.get(DATA+'/v2/trades',params=params,timeout=30)
                    response.raise_for_status();payload=response.json()
                    if not isinstance(payload,dict) or not isinstance(payload.get('data'),list):
                        raise ValueError('Unexpected trade endpoint schema')
                except Exception as e:
                    failures.append({'condition_id':market_id,'page':pages,'error':str(e)[:180]});break
                for t in payload['data']:
                    row={'condition_id':str(t.get('condition_id') or t.get('conditionId') or market_id),
                         'proxy_wallet':t.get('proxy_wallet') or t.get('proxyWallet'),
                         'outcome':t.get('outcome'),'side':t.get('side'),'price':t.get('price'),
                         'size':t.get('size'),'timestamp':t.get('timestamp'),
                         'transaction_hash':t.get('transaction_hash') or t.get('transactionHash')}
                    # Do not deduplicate repeated-looking fills: every v2 taker fill has its own meaning.
                    if row['condition_id']!=market_id:continue
                    writer.writerow(row);total+=1
                pages+=1
                nxt=(payload.get('pagination') or {}).get('next_cursor') or payload.get('next_cursor')
                if not nxt:break
                if nxt in seen:
                    failures.append({'condition_id':market_id,'page':pages,'error':'Repeated cursor / pagination loop'});break
                seen.add(nxt);cursor=nxt
                if delay:time.sleep(delay)
            condition_page_counts[market_id]=pages
            if pages>=max_pages_per_market and cursor:incomplete.append(market_id)
            f.flush()
    report={'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
        'markets_requested':len(markets),'fills_written':total,
        'page_counts':condition_page_counts,'markets_truncated_by_page_limit':incomplete,
        'request_failures':failures,
        'requires_review_before_training':bool(failures or incomplete),
        'description':'API taker fills; historical resolution labels and freeze times must come from independent metadata.'}
    dest.with_suffix('.download_report.json').write_text(json.dumps(report,indent=2))
    return report
