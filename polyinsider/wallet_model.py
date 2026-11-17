"""Chronologically honest wallet-outperformance classifier.

Prediction target: positive mean signed price advantage in LATER resolved events.
It is NOT a classifier of possession of nonpublic information.

A training example is frozen before future events: features use ONLY fully
resolved earlier event groups; its label is derived from a disjoint later window.
Windows A, B fit model; untouched C evaluates it. Related outcomes aggregate at
event-wallet level. Multiple examples of a wallet in the fit are acceptable:
window chronology, not wallet independence, is the estimand. Confidence intervals
must be clustered by wallet/event before claims of statistical significance.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from .core import normalize_history, to_utc

WALLET_FEATURES=['prior_event_count_log','prior_mean_edge','prior_positive_share',
                 'prior_edge_standardized','prior_avg_stake_log']

def event_wallet_observations(history):
    """One equal-event-weight signed price advantage per event/wallet.

    An entry is only eligible AFTER every market/outcome in its event resolved.
    A buy-direction sign is a proxy; it does not reconstruct wallet positions.
    """
    d=normalize_history(history)
    event_ends=d.groupby('event_id').resolution_time.max().rename('event_end')
    d['edge']=np.where(d.direction=='YES',d.resolved_yes-d.yes_price,d.yes_price-d.resolved_yes)
    grouped=d.groupby(['event_id','wallet'],as_index=False).agg(edge=('edge','mean'),
                          avg_stake=('amount_usd','mean'),markets=('condition_id','nunique'))
    grouped=grouped.merge(event_ends,on='event_id',validate='many_to_one')
    return grouped.sort_values(['event_end','event_id','wallet']).reset_index(drop=True)

def prior_wallet_features(rows,freeze_at,min_prior=3):
    at=to_utc(freeze_at)
    old=rows.loc[rows.event_end<at]
    if old.empty:return pd.DataFrame(columns=['wallet','prior_events',*WALLET_FEATURES,'prior_mean_edge_raw'])
    g=old.groupby('wallet').agg(n=('edge','size'),mean_edge=('edge','mean'),
           positive=('edge',lambda s:float((s>0).mean())),avg_stake=('avg_stake','mean'))
    g=g[g.n>=min_prior].copy()
    g['prior_event_count_log']=np.log1p(g.n)
    g['prior_mean_edge']=g.mean_edge
    g['prior_positive_share']=g.positive
    g['prior_edge_standardized']=g.mean_edge*np.sqrt(g.n)   # bounded signed edge, *not* z-score
    g['prior_avg_stake_log']=np.log1p(g.avg_stake)
    g['prior_mean_edge_raw']=g.mean_edge
    g['prior_events']=g.n.astype(int)
    return g.reset_index()[['wallet','prior_events',*WALLET_FEATURES,'prior_mean_edge_raw']]

def training_window(rows,freeze_at,label_until,min_prior=3,min_future=3):
    """Excludes boundary events and requires future observations for supervised labels."""
    cut=to_utc(freeze_at);end=to_utc(label_until)
    if not cut<end: raise ValueError('Label window end must follow feature cutoff')
    f=prior_wallet_features(rows,cut,min_prior=min_prior)
    future=rows[(rows.event_end>=cut)&(rows.event_end<end)]
    if future.empty:return f.iloc[:0].copy()
    labels=future.groupby('wallet').agg(future_events=('edge','size'),future_mean_edge=('edge','mean'))
    labels=labels[labels.future_events>=min_future]
    out=f.merge(labels.reset_index(),on='wallet',how='inner')
    out['future_outperform']= (out.future_mean_edge>0).astype(int)
    out['features_frozen_at']=cut.isoformat();out['labels_end_before']=end.isoformat()
    return out

def prob_wallet(model,features):
    """Pure inference from already frozen features; no future labels here."""
    xs=np.array([float(features[k]) for k in WALLET_FEATURES])
    scaled=(xs-np.array(model['feature_means']))/np.array(model['feature_scales'])
    z=float(model['intercept'])+float(np.dot(np.array(model['coefficients']),scaled))
    return float(1/(1+math.exp(-max(-35,min(35,z)))))

def train_wallet_outperformance(history,source='UPLOADED DATA — UNVERIFIED',asof=None,min_prior=3,min_future=3):
    d=normalize_history(history)
    if asof is not None:
        asof=to_utc(asof)
    else:asof=pd.Timestamp.now(tz='UTC')
    # Future outcomes are disallowed, even if CSV contains accidentally pre-labeled markets.
    complete_events=d.groupby('event_id').resolution_time.max()
    eligible_events=set(complete_events[complete_events<asof].index)
    d=d[d.event_id.isin(eligible_events)].copy()
    if d.empty:raise ValueError('No historically resolved events strictly before asof')
    # Drop incomplete event group when any of its outcomes resolve after asof.
    # Supplier must supply complete event outcome groups; the grouping is only as complete as the input.
    rows=event_wallet_observations(d)
    times=rows.groupby('event_id').event_end.max().sort_values().drop_duplicates()
    if len(times)<24:raise ValueError('Need at least 24 distinct event-resolution times for 3 temporal windows')
    t=times.reset_index(drop=True)
    c1=t.iloc[int(len(t)*.35)];c2=t.iloc[int(len(t)*.55)];c3=t.iloc[int(len(t)*.77)]
    # Unique timestamp quantiles and disjoint label intervals.
    if not c1<c2<c3:raise ValueError('Insufficient distinct event-resolution cutoffs')
    # Last window exclusive end: one nanosecond after last event timestamp.
    c4=t.iloc[-1]+pd.Timedelta(nanoseconds=1)
    a=training_window(rows,c1,c2,min_prior,min_future)
    b=training_window(rows,c2,c3,min_prior,min_future)
    test=training_window(rows,c3,c4,min_prior,min_future)
    fit=pd.concat([a,b],ignore_index=True)
    if len(fit)<25 or len(test)<8:
        raise ValueError(f'Insufficient wallet-window examples: fit={len(fit)} test={len(test)}. Supply more full history.')
    if fit.future_outperform.nunique()!=2 or test.future_outperform.nunique()!=2:
        raise ValueError('Need both labels in fit and test windows')
    scaler=StandardScaler().fit(fit[WALLET_FEATURES])
    clf=LogisticRegression(max_iter=1000,C=.5,random_state=473).fit(
        scaler.transform(fit[WALLET_FEATURES]),fit.future_outperform)
    probs=clf.predict_proba(scaler.transform(test[WALLET_FEATURES]))[:,1]
    base= float(fit.future_outperform.mean())
    report={'source':source,'target':'positive mean signed price edge on >=3 future resolved events (NOT private information)',
       'train_wallet_windows':int(len(fit)),'test_wallets':int(len(test)),
       'train_unique_wallets':int(fit.wallet.nunique()),'test_unique_wallets':int(test.wallet.nunique()),
       'train_labels_end_before':c3.isoformat(),'test_features_frozen_at':c3.isoformat(),
       'test_labels_end_before':c4.isoformat(),
       'test_auc':round(float(roc_auc_score(test.future_outperform,probs)),4),
       'test_brier':round(float(brier_score_loss(test.future_outperform,probs)),4),
       'base_rate_brier':round(float(brier_score_loss(test.future_outperform,np.repeat(base,len(test)))),4),
       'test_positive_share':round(float(test.future_outperform.mean()),4),
       'test_example':test[['wallet','prior_events','future_events','future_mean_edge','future_outperform']].head(5).to_dict('records'),
       'limitations':'One repeated wallet can appear across time windows; sample does not independently prove insider knowledge or warrant claims of significance.'}
    # Fit final model on all measured labels, so live inference can use all resolved history.
    all_examples=pd.concat([fit,test],ignore_index=True)
    finalsc=StandardScaler().fit(all_examples[WALLET_FEATURES])
    final=LogisticRegression(max_iter=1000,C=.5,random_state=473).fit(
       finalsc.transform(all_examples[WALLET_FEATURES]),all_examples.future_outperform)
    model={'kind':'future_wallet_outperformance_probability',
       'target':report['target'],'source':source,'trained_at':datetime.now(timezone.utc).isoformat(),
       'asof':asof.isoformat(),'last_training_resolution':rows.event_end.max().isoformat(),
       'features':WALLET_FEATURES,'intercept':float(final.intercept_[0]),
       'coefficients':list(map(float,final.coef_[0])),
       'feature_means':list(map(float,finalsc.mean_)),
       'feature_scales':list(map(float,finalsc.scale_)),
       'minimum_prior_events':int(min_prior),'minimum_future_events':int(min_future),
       'trained_examples':len(all_examples)}
    state=prior_wallet_features(rows,asof,min_prior).copy()
    state['prob_future_outperform']=state.apply(lambda r:prob_wallet(model,r),axis=1)
    return model,report,state

def train_wallet_file(csv_path,out_dir,source=None,asof=None,min_prior=3,min_future=3):
    src=source or ('SYNTHETIC DEMONSTRATION' if 'synthetic' in str(csv_path).lower() else 'USER-PROVIDED CSV — UNVERIFIED')
    m,r,s=train_wallet_outperformance(pd.read_csv(csv_path),src,asof,min_prior,min_future)
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'wallet_model.json').write_text(json.dumps(m,indent=2))
    (out/'wallet_backtest.json').write_text(json.dumps(r,indent=2))
    s.to_csv(out/'wallet_probabilities.csv',index=False)
    return r
