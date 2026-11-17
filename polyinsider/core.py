"""Strict as-of wallet features, chronological model fitting, and audit-friendly evaluation.

Research-grade prototype, not a private information or trading detector.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

COLUMNS = ['event_id','condition_id','title','snapshot_time','resolution_time',
           'yes_price','wallet','direction','amount_usd','resolved_yes']
FEATURES = ['logit_price','net_flow','skill_flow','wallet_count','log_volume']

def to_utc(x):
    dt = pd.Timestamp(x)
    if pd.isna(dt): raise ValueError('Missing timestamp')
    return dt.tz_localize('UTC') if dt.tzinfo is None else dt.tz_convert('UTC')

def normalize_history(df):
    missing = sorted(set(COLUMNS)-set(df.columns))
    if missing: raise ValueError(f'Historical CSV missing columns: {missing}')
    d=df[COLUMNS].copy()
    for k in ['snapshot_time','resolution_time']:
        d[k]=pd.to_datetime(d[k], utc=True, errors='raise', format='ISO8601')
    for k in ['yes_price','amount_usd','resolved_yes']:
        d[k]=pd.to_numeric(d[k],errors='raise')
    if d.empty: raise ValueError('Empty historical dataset')
    if not d.yes_price.between(.001,.999).all():raise ValueError('Prices must be in (.001, .999)')
    if not d.resolved_yes.isin([0,1]).all():raise ValueError('resolved_yes must be 0 or 1')
    if not (d.snapshot_time<d.resolution_time).all():raise ValueError('Every observation must precede resolution')
    if not d.direction.str.upper().isin(['YES','NO']).all():raise ValueError('Direction must be YES or NO')
    if not (d.amount_usd>0).all():raise ValueError('amount_usd must be positive')
    for id_, group in d.groupby('condition_id'):
        if group[['snapshot_time','resolution_time','yes_price','resolved_yes','event_id']].nunique().max()>1:
            raise ValueError(f'One frozen snapshot per condition, conflicting metadata: {id_}')
    d['wallet']=d.wallet.astype(str).str.lower()
    d['direction']=d.direction.str.upper()
    return d.sort_values('snapshot_time').reset_index(drop=True)

def market_features(price, trades, wallet_stats):
    """Trades: iterable of {wallet,direction,amount_usd}. No outcome inputs."""
    p=float(np.clip(price,.001,.999)); buy=0.; total=0.; weighted=0.; ws=set()
    for trade in trades:
        amt=max(float(trade['amount_usd']),0.)
        if not amt: continue
        side=1. if str(trade['direction']).upper()=='YES' else -1.
        wallet=str(trade['wallet']).lower(); ws.add(wallet)
        n, s = wallet_stats.get(wallet,(0,0.0))
        skill=float(np.clip(s/math.sqrt(max(n,1)), -2.0, 2.0)) if n>=3 else 0.
        buy+=side*amt; weighted+=side*amt*skill; total+=amt
    return {'logit_price':float(np.log(p/(1-p))),
            'net_flow':float(buy/total) if total else 0.,
            'skill_flow':float(weighted/total) if total else 0.,
            'wallet_count':float(np.log1p(len(ws))),
            'log_volume':float(np.log1p(total))}

def wallet_signals(snapshot, wallet_stats):
    ans=[]
    for t in snapshot:
        wallet=str(t['wallet']).lower()
        n,s=wallet_stats.get(wallet,(0,0.))
        score=s/math.sqrt(n) if n>0 else 0.
        ans.append({'wallet':wallet,'prior_events':int(n),'prior_score':round(float(score),3),
                    'direction':t['direction'],'amount_usd':round(float(t['amount_usd']),2),
                    'history_flag': bool(n>=10 and score>=1.5)})
    return sorted(ans,key=lambda x:(-x['prior_score'],-x['amount_usd']))

def chronological_features(history):
    """At each snapshot, only event labels publicly resolved before the snapshot enter wallet history.
    Whole events become eligible together at their latest resolution timestamp, preventing sibling-outcome leakage.
    """
    d=normalize_history(history)
    event_end = d.groupby('event_id').resolution_time.max().to_dict()
    event_contrib={}
    for event, rows in d.groupby('event_id'):
        by_wallet=defaultdict(list)
        for _,r in rows.iterrows():
            edge = (r.resolved_yes-r.yes_price) if r.direction=='YES' else (r.yes_price-r.resolved_yes)
            by_wallet[r.wallet].append(float(edge))
        event_contrib[event] = {w:float(np.mean(v)) for w,v in by_wallet.items()}
    releases=sorted(event_end.items(),key=lambda x:x[1]); release_idx=0
    wallet_stats=defaultdict(lambda:(0,0.0))
    rows_out=[]
    for condition,group in d.groupby('condition_id',sort=False):
        time=group.snapshot_time.iloc[0]
        while release_idx<len(releases) and releases[release_idx][1] < time:
            event_id,_=releases[release_idx]
            for w,edge in event_contrib[event_id].items():
                n,s=wallet_stats[w];wallet_stats[w]=(n+1,s+edge)
            release_idx+=1
        sample=[{'wallet':r.wallet,'direction':r.direction,'amount_usd':r.amount_usd}
                for _,r in group.iterrows()]
        features=market_features(float(group.yes_price.iloc[0]),sample,wallet_stats)
        r=group.iloc[0]
        rows_out.append({'event_id':r.event_id,'condition_id':condition,'title':r.title,
              'snapshot_time':time,'resolution_time':r.resolution_time,'resolved_yes':int(r.resolved_yes),
              'yes_price':float(r.yes_price),**features})
    return pd.DataFrame(rows_out).sort_values('snapshot_time').reset_index(drop=True)

def wallet_state_asof(history, asof):
    """Construct wallet performance from complete events resolved before `asof`, not from current/open markets."""
    d=normalize_history(history); date=to_utc(asof)
    events=d.groupby('event_id').resolution_time.max()
    available=set(events[events<date].index)
    stats=defaultdict(lambda:[0,0.]); event_summaries=defaultdict(lambda:defaultdict(list))
    for _,r in d[d.event_id.isin(available)].iterrows():
        edge=(r.resolved_yes-r.yes_price) if r.direction=='YES' else (r.yes_price-r.resolved_yes)
        event_summaries[r.event_id][r.wallet].append(float(edge))
    for ev, groups in event_summaries.items():
        for wallet,edges in groups.items():
            stats[wallet][0]+=1; stats[wallet][1]+=float(np.mean(edges))
    return {w:(int(v[0]),float(v[1])) for w,v in stats.items()}

def expit(v):
    return float(1/(1+np.exp(-np.clip(v,-35,35))))

def model_probability(model,features):
    return expit(float(model['intercept'])+sum(float(model['coefficients'][k])*float(features[k]) for k in FEATURES))

def metrics(y,p):
    y=np.array(y,dtype=int); p=np.clip(np.array(p,dtype=float),.001,.999)
    return {'n':int(len(y)),'brier':round(float(brier_score_loss(y,p)),4),
        'log_loss':round(float(log_loss(y,p,labels=[0,1])),4),
        'auc':round(float(roc_auc_score(y,p)),4) if len(set(y))>1 else None}

def train_evaluate(history, source='UPLOADED HISTORICAL DATA', cutoff=None):
    d=normalize_history(history)
    f=chronological_features(d)
    if len(f)<30: raise ValueError('Need at least 30 markets for a defensible initial train/test split')
    if cutoff is not None:
        cut=to_utc(cutoff)
        eligible=f[f.resolution_time<cut].copy()
    else: eligible=f.copy()
    if len(eligible)<30:raise ValueError('Need >=30 fully resolved markets before training cutoff')
    # Split by *event*, not by market. Training labels must be resolved before the first test feature snapshot.
    events=eligible.groupby('event_id').agg(end=('resolution_time','max'),first=('snapshot_time','min')).sort_values('end')
    candidate_idx=max(15,int(len(events)*.7))
    chosen=None
    for i in range(candidate_idx,len(events)-5):
        test_start=events.iloc[i:]['first'].min()
        train_evs=set(events.iloc[:i].loc[events.iloc[:i]['end'] < test_start].index)
        test_evs=set(events.iloc[i:].index)
        train_df=eligible[eligible.event_id.isin(train_evs)]
        test_df=eligible[eligible.event_id.isin(test_evs)]
        if len(train_df)>=15 and len(test_df)>=6 and train_df.resolved_yes.nunique()==2 and test_df.resolved_yes.nunique()==2:
            chosen=(train_df,test_df,test_start);break
    if chosen is None:raise ValueError('No valid temporal train/test split; check overlapping event dates and class balance')
    train_df,test_df,test_start=chosen
    xtrain=train_df[FEATURES].to_numpy(float); ytrain=train_df.resolved_yes.astype(int)
    xtest=test_df[FEATURES].to_numpy(float); ytest=test_df.resolved_yes.astype(int)
    clf=LogisticRegression(max_iter=2000,C=.5,random_state=473)
    clf.fit(xtrain,ytrain)
    testp=clf.predict_proba(xtest)[:,1]
    test_report={'trained_markets':len(train_df),'test_markets':len(test_df),
        'train_events':int(train_df.event_id.nunique()),'test_events':int(test_df.event_id.nunique()),
        'train_last_resolution':train_df.resolution_time.max().isoformat(),
        'test_first_snapshot':test_start.isoformat(),
        'model':metrics(ytest,testp),'market_baseline':metrics(ytest,test_df.yes_price),
        'note':'Held-out events only; no training event resolves after first test snapshot.'}
    # Final model for snapshots TODAY is fitted on all labels already resolved by cutoff.
    final=LogisticRegression(max_iter=2000,C=.5,random_state=473)
    final.fit(eligible[FEATURES].to_numpy(float),eligible.resolved_yes.astype(int))
    model={'kind':'pre_resolution_event_yes_forecast','trained_at':datetime.now(timezone.utc).isoformat(),
       'training_cutoff':cutoff or 'all resolved rows supplied',
       'source':source,'features':FEATURES,
       'intercept':float(final.intercept_[0]),
       'coefficients':dict(zip(FEATURES,map(float,final.coef_[0]))),
       'number_of_training_markets':int(len(eligible)),
       'number_of_training_events':int(eligible.event_id.nunique())}
    state=wallet_state_asof(d, cut if cutoff is not None else pd.Timestamp.now(tz='UTC'))
    return model,state,test_report,f

def write_training(history_path, output_dir, source=None,cutoff=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(history_path)
    label=source or ('SYNTHETIC DEMONSTRATION' if 'synthetic' in str(history_path).lower() else 'USER-PROVIDED HISTORICAL CSV — NOT AUDITED')
    model,state,report,features=train_evaluate(df,source=label,cutoff=cutoff)
    (out/'model.json').write_text(json.dumps(model,indent=2))
    (out/'wallet_state.json').write_text(json.dumps({k:list(v) for k,v in state.items()},indent=2))
    (out/'backtest.json').write_text(json.dumps({'source':label,**report},indent=2))
    features.to_csv(out/'asof_features.csv',index=False)
    return report
