#!/usr/bin/env python3
"""ECE 473 | Paired, chronological historical ablation of wallet-skill flow.

Usage (from the repository root):
    python test_wallet_skill_contribution.py --csv data/prepared_history.csv

This script NEVER overwrites data/run or existing live models; outputs to
`data/wallet_skill_ablation` unless --out is provided.

Research limitations: uses the same frozen historical fill-price proxy and
wallet-score features as polyinsider.core; does not validate information release
or settlement times; confidence intervals condition on fitted models.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from polyinsider.core import FEATURES, chronological_features, to_utc


def train_test_split(f, cutoff=None):
    """Exact event-based chronological split used in polyinsider.core.train_evaluate."""
    if cutoff:
        f=f.loc[f.resolution_time < to_utc(cutoff)].copy()
    if len(f)<30:
        raise ValueError('Need at least 30 resolved markets')
    events=(f.groupby('event_id')
              .agg(end=('resolution_time','max'), first=('snapshot_time','min'))
              .sort_values('end'))
    candidate_idx=max(15,int(len(events)*.7))
    for i in range(candidate_idx, len(events)-5):
        test_start=events.iloc[i:]['first'].min()
        train_events=set(events.iloc[:i].loc[events.iloc[:i]['end'] < test_start].index)
        test_events=set(events.iloc[i:].index)
        train=f[f.event_id.isin(train_events)].copy()
        test=f[f.event_id.isin(test_events)].copy()
        if (len(train)>=15 and len(test)>=6 and
            train.resolved_yes.nunique()==2 and test.resolved_yes.nunique()==2):
            if not train.resolution_time.max() < test.snapshot_time.min():
                raise AssertionError('Temporal leakage: train label after test snapshot')
            if set(train.event_id).intersection(test.event_id):
                raise AssertionError('Event leakage across folds')
            return train, test
    raise ValueError('No valid split, inspect event dates and class balance')


def fit_predict(train, test, columns):
    reg=LogisticRegression(max_iter=2000,C=.5,random_state=473)
    reg.fit(train[columns].to_numpy(float),train.resolved_yes.to_numpy(int))
    proba=reg.predict_proba(test[columns].to_numpy(float))[:,1]
    return np.clip(proba, .001, .999),reg


def measures(y, pred):
    return {
        'n':int(len(y)),
        'brier':round(float(brier_score_loss(y,pred)),6),
        'log_loss':round(float(log_loss(y,pred,labels=[0,1])),6),
        'auc':round(float(roc_auc_score(y,pred)),6) if len(np.unique(y))==2 else None
    }


def bootstrap_events(test, predictions, samples=1000, seed=473):
    """Paired cluster bootstrap of the test metrics, treating events as clusters.

    CIs condition on already-fitted models and on the observed historical sample;
    they do not capture data-source or training-set uncertainty.
    """
    ids=test.event_id.to_numpy()
    group_ids=np.unique(ids)
    indices={eid:np.flatnonzero(ids==eid) for eid in group_ids}
    rng=np.random.default_rng(seed)
    y=test.resolved_yes.to_numpy(dtype=int)
    brier_delta=[]
    auc_delta=[]
    for _ in range(samples):
        picked=rng.choice(group_ids,size=len(group_ids),replace=True)
        idx=np.concatenate([indices[eid] for eid in picked])
        yy=y[idx]
        full=predictions['with_skill'][idx]
        no_skill=predictions['without_skill'][idx]
        brier_delta.append(float(np.mean((yy-full)**2-(yy-no_skill)**2)))
        if np.unique(yy).size==2:
            auc_delta.append(float(roc_auc_score(yy,full)-roc_auc_score(yy,no_skill)))
    def ci(a):
        return [round(float(t),6) for t in np.quantile(a,[.025,.975])] if a else None
    return {'n_bootstrap':samples,'n_event_clusters':int(len(group_ids)),
            'delta_brier_ci95':ci(brier_delta),
            'delta_auc_ci95':ci(auc_delta),
            'interpretation':'Delta Brier = with skill MINUS without skill (negative favors skill). Delta AUC = with skill MINUS without skill (positive favors skill).',
            'caveat':'Event-clustered, paired, test-set bootstrap conditional on trained models; not a prospective trial or a full training-set bootstrap.'}


def main():
    ap=argparse.ArgumentParser(description='Out-of-sample wallet-skill ablation with matched splits')
    ap.add_argument('--csv',default='data/prepared_history.csv')
    ap.add_argument('--out',default='data/wallet_skill_ablation')
    ap.add_argument('--cutoff',default=None,help='Optional ISO 8601 cutoff on resolved training/history rows')
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    if args.bootstrap<0:ap.error('--bootstrap cannot be negative')
    src=Path(args.csv)
    if not src.exists():ap.error(f'Cannot find CSV: {src}')
    print('Reading historical data (read-only):',src,flush=True)
    hist=pd.read_csv(src)
    f=chronological_features(hist)
    train,test=train_test_split(f,cutoff=args.cutoff)
    no_skill=[c for c in FEATURES if c!='skill_flow']
    y=test.resolved_yes.to_numpy(int)
    preds={}
    for name,cols in [
        ('with_skill',FEATURES),
        ('without_skill',no_skill),
        ('price_only_recalibrated',['logit_price'])]:
        preds[name],model=fit_predict(train,test,cols)
    preds['raw_market_price']=np.clip(test.yes_price.to_numpy(float),.001,.999)
    report={
        'generated_at':datetime.now(timezone.utc).isoformat(),
        'source_csv':str(src),
        'analysis':'Same training and test markets, chronological event-disjoint split, logistic regression with C=0.5 and random_state=473',
        'training_markets':int(len(train)),
        'training_events':int(train.event_id.nunique()),
        'test_markets':int(len(test)),
        'test_events':int(test.event_id.nunique()),
        'last_training_resolution':train.resolution_time.max().isoformat(),
        'first_test_snapshot':test.snapshot_time.min().isoformat(),
        'test_markets_with_nonzero_skill_flow':int((test.skill_flow.abs()>1e-12).sum()),
        'test_markets_with_zero_skill_flow':int((test.skill_flow.abs()<=1e-12).sum()),
        'metrics':{name:measures(y,p) for name,p in preds.items()},
        'delta_brier_full_minus_without':round(float(np.mean((y-preds['with_skill'])**2-(y-preds['without_skill'])**2)),6),
        'delta_auc_full_minus_without':round(float(roc_auc_score(y,preds['with_skill'])-roc_auc_score(y,preds['without_skill'])),6),
        'mean_absolute_prediction_change_with_skill':round(float(np.mean(np.abs(preds['with_skill']-preds['without_skill']))),6),
        'bootstrap':bootstrap_events(test,preds,args.bootstrap) if args.bootstrap else None,
        'limitations':[
            'Historical trade prices and signed flow are proxies; they are not full order books or a complete PnL ledger.',
            'The event info-time and resolution-time metadata have not been independently audited.',
            'This tests incremental value of a skill_flow feature, NOT possession of nonpublic information.',
            'Repeated markets inside one event can be correlated; bootstrap resamples whole events.',
            'The final live market model should NOT be automatically replaced by either ablation model.'
        ]
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    export=test[['event_id','condition_id','title','snapshot_time','resolution_time','resolved_yes','yes_price','skill_flow']].copy()
    for key,pred in preds.items():export['p_'+key]=pred
    export.to_csv(out/'paired_predictions.csv',index=False)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('\n--- WALLET SKILL ABLATION (HELD-OUT HISTORICAL TEST) ---')
    print(f"Training: {report['training_markets']} markets / {report['training_events']} events")
    print(f"Test: {report['test_markets']} markets / {report['test_events']} events")
    print(f"Test markets with nonzero skill_flow: {report['test_markets_with_nonzero_skill_flow']}")
    print(f"Last training resolution: {report['last_training_resolution']}")
    print(f"First test snapshot: {report['first_test_snapshot']}")
    print('')
    print(f"{'MODEL':<28} {'BRIER':>10} {'AUC':>10} {'LOG LOSS':>10}")
    for name,item in report['metrics'].items():
        print(f"{name:<28} {item['brier']:>10.4f} {item['auc']:>10.4f} {item['log_loss']:>10.4f}")
    print('\nDelta Brier (with minus without):',report['delta_brier_full_minus_without'])
    print('Delta AUC   (with minus without):',report['delta_auc_full_minus_without'])
    if report['bootstrap']:
        print('Event-bootstrap 95% CI, delta Brier:',report['bootstrap']['delta_brier_ci95'])
        print('Event-bootstrap 95% CI, delta AUC:',report['bootstrap']['delta_auc_ci95'])
    print('\nResults:',out/'results.json')
    print('Paired predictions:',out/'paired_predictions.csv')
    print('No changes made to data/run or the live website.')

if __name__=='__main__':main()
