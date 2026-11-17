#!/usr/bin/env python3
"""ECE 473 | Leakage-aware monthly walk-forward market prediction evaluation.

From PolyInsiders_ECE473_Updated root:
    python test_walk_forward_polyinsiders.py --features data/run/asof_features.csv

Read-only input, outputs to data/walk_forward_evaluation by default.
This is HISTORICAL backtesting, not prospective validation or proof of insider information.
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

FEATURES = ['logit_price','net_flow','skill_flow','wallet_count','log_volume']
ALTERNATIVES = {
    'with_skill': FEATURES,
    'without_skill': [c for c in FEATURES if c != 'skill_flow'],
    'price_only_recalibrated': ['logit_price'],
}


def summarize(y, p):
    y = np.asarray(y, int)
    p = np.clip(np.asarray(p, float), .001, .999)
    if len(y) == 0:
        return None
    return dict(n=int(len(y)), brier=round(float(brier_score_loss(y, p)), 6),
                log_loss=round(float(log_loss(y, p, labels=[0,1])), 6),
                auc=(round(float(roc_auc_score(y, p)), 6) if len(np.unique(y)) > 1 else None))


def bootstrap_event_delta(pairs, n_bootstrap=500, seed=473):
    """Event-clustered paired uncertainty for OOS prediction deltas; fits held fixed."""
    if n_bootstrap <= 0:
        return None
    ids = pairs.event_id.astype(str).to_numpy()
    unique = np.unique(ids)
    lookup = {e:np.flatnonzero(ids == e) for e in unique}
    rng = np.random.default_rng(seed)
    y = pairs.resolved_yes.to_numpy(int)
    full = pairs.p_with_skill.to_numpy(float)
    reduced = pairs.p_without_skill.to_numpy(float)
    market = pairs.p_raw_market_price.to_numpy(float)
    d_brier, d_vs_market, d_auc = [], [], []
    for _ in range(n_bootstrap):
        selected = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([lookup[e] for e in selected])
        yy = y[index]
        d_brier.append(float(np.mean((yy-full[index])**2 - (yy-reduced[index])**2)))
        d_vs_market.append(float(np.mean((yy-full[index])**2 - (yy-market[index])**2)))
        if len(np.unique(yy)) > 1:
            d_auc.append(float(roc_auc_score(yy,full[index])-roc_auc_score(yy,reduced[index])))
    interval = lambda arr: [round(float(v), 7) for v in np.quantile(arr, [.025,.975])] if arr else None
    return {'resamples':n_bootstrap, 'event_clusters':len(unique),
            'brier_with_minus_without_ci95':interval(d_brier),
            'brier_with_minus_raw_market_ci95':interval(d_vs_market),
            'auc_with_minus_without_ci95':interval(d_auc),
            'caveat':'Resamples evaluation events, conditions on fitted models; not an independent future validation.'}


def run_walk_forward(f, min_train_markets=20, min_test_markets=8, n_bootstrap=500):
    f = f.copy()
    required = set(['event_id','condition_id','snapshot_time','resolution_time',
                    'resolved_yes','yes_price',*FEATURES])
    missing = sorted(required-set(f.columns))
    if missing: raise ValueError(f'Missing columns: {missing}')
    f['event_id'] = f.event_id.astype(str)
    f['condition_id'] = f.condition_id.astype(str)
    for key in ['snapshot_time','resolution_time']:
        f[key] = pd.to_datetime(f[key], utc=True, errors='raise', format='ISO8601')
    for key in ['resolved_yes','yes_price',*FEATURES]:
        f[key] = pd.to_numeric(f[key],errors='raise')
    if f[list(required.difference(['event_id','condition_id','snapshot_time','resolution_time']))].isna().any().any():
        raise ValueError('Missing numeric model data')
    if f.condition_id.duplicated().any():
        raise ValueError('Expected one frozen snapshot per market; duplicate condition_id found')
    if not (f.snapshot_time < f.resolution_time).all():
        raise ValueError('Found a prediction snapshot on or after its market resolution')
    if not f.resolved_yes.isin([0,1]).all():
        raise ValueError('Target must be binary')
    if not f.yes_price.between(.001,.999).all():
        raise ValueError('Market probabilities must be between .001 and .999 inclusive')
    f = f.sort_values('snapshot_time').reset_index(drop=True)
    event_end = f.groupby('event_id').resolution_time.max()
    starts = pd.date_range(start=f.snapshot_time.min().normalize().replace(day=1),
                           end=f.snapshot_time.max().normalize().replace(day=1),
                           freq='MS', tz='UTC')
    folds = []
    pred_frames=[]
    skips=[]
    for start in starts:
        end = start + pd.DateOffset(months=1)
        historical_events = set(event_end[event_end < start].index)
        train = f[f.event_id.isin(historical_events)].copy()
        test = f[(f.snapshot_time >= start) & (f.snapshot_time < end) &
                 (~f.event_id.isin(historical_events))].copy()
        if (len(train) < min_train_markets or len(test) < min_test_markets or
            train.resolved_yes.nunique() != 2):
            skips.append({'window':start.strftime('%Y-%m'), 'train_markets':len(train),
                          'test_markets':len(test), 'reason':'insufficient history or test markets or train class balance'})
            continue
        if len(set(train.event_id)&set(test.event_id)):
            raise AssertionError('Training/test event overlap')
        if not train.resolution_time.max() < test.snapshot_time.min():
            raise AssertionError('Training labels were not available before test snapshots')
        y = test.resolved_yes.to_numpy(int)
        values = {'raw_market_price':np.clip(test.yes_price.to_numpy(float), .001,.999)}
        for name, cols in ALTERNATIVES.items():
            model = LogisticRegression(max_iter=2000,C=.5,random_state=473)
            model.fit(train[cols].to_numpy(float),train.resolved_yes.to_numpy(int))
            values[name] = np.clip(model.predict_proba(test[cols].to_numpy(float))[:,1], .001,.999)
        out = test[['event_id','condition_id','snapshot_time','resolution_time','resolved_yes','yes_price','skill_flow']].copy()
        out['fold'] = start.strftime('%Y-%m')
        for name,p in values.items():
            out[f'p_{name}'] = p
        folds.append({'window':start.strftime('%Y-%m'), 'train_events':int(train.event_id.nunique()),
                      'train_markets':len(train), 'test_events':int(test.event_id.nunique()),
                      'test_markets':len(test),
                      'last_train_resolution':train.resolution_time.max().isoformat(),
                      'first_test_snapshot':test.snapshot_time.min().isoformat(),
                      'with_skill':summarize(y,values['with_skill']),
                      'without_skill':summarize(y,values['without_skill']),
                      'raw_market_price':summarize(y,values['raw_market_price'])})
        pred_frames.append(out)

    if not pred_frames:
        raise ValueError('No evaluable monthly folds. Check timestamp distribution and sample sizes.')
    pairs = pd.concat(pred_frames,ignore_index=True)
    if pairs.condition_id.duplicated().any():
        raise AssertionError('One market predicted in multiple folds')
    y=pairs.resolved_yes.to_numpy(int)
    metrics={name:summarize(y,pairs[f'p_{name}']) for name in [*ALTERNATIVES,'raw_market_price']}
    delta_brier=float(np.mean((y-pairs.p_with_skill.to_numpy())**2-(y-pairs.p_without_skill.to_numpy())**2))
    auc_full=metrics['with_skill']['auc']; auc_other=metrics['without_skill']['auc']
    result={'generated_at':datetime.now(timezone.utc).isoformat(),
        'method':'Monthly expanding-window evaluation: train only on fully resolved prior events; test markets snapshotted in next calendar month; refit each month; event-disjoint folds.',
        'prepared_markets':len(f),'prepared_events':int(f.event_id.nunique()),
        'evaluated_markets':len(pairs),'evaluated_events':int(pairs.event_id.nunique()),
        'unevaluated_markets':len(f)-len(pairs),'n_monthly_folds':len(folds),
        'metrics':metrics, 'brier_with_minus_without':round(delta_brier,7),
        'auc_with_minus_without':round(auc_full-auc_other,7) if auc_full is not None and auc_other is not None else None,
        'bootstrap':bootstrap_event_delta(pairs,n_bootstrap),
        'folds':folds,'skipped_windows':skips,
        'limitations':[
           'Past source data use estimated public information times and prices; timing/settlement need independent audit.',
           'This is an observational historical backtest, not a prospective trial or evidence of nonpublic information.',
           'Training months must have sufficient already-resolved history; early months may be skipped.',
           'Several markets can share an event, so confidence intervals resample events, not rows.',
           'Model feature construction is inherited from the previously computed asof_features.csv.',
           'This script does not modify or deploy data/run models or live web output.'
        ]}
    return result,pairs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',default='data/run/asof_features.csv')
    p.add_argument('--out',default='data/walk_forward_evaluation')
    p.add_argument('--min-train',type=int,default=20)
    p.add_argument('--min-test',type=int,default=8)
    p.add_argument('--bootstrap',type=int,default=500)
    args=p.parse_args()
    source=Path(args.features)
    if not source.is_file(): p.error(f'Missing as-of features file: {source}')
    print('Reading as-of historical features (read only):',source,flush=True)
    f=pd.read_csv(source)
    report,predictions=run_walk_forward(f,args.min_train,args.min_test,args.bootstrap)
    report['input_features_path']=str(source)
    model_meta=source.parent/'model.json'
    if model_meta.is_file():
        try:
            report['model_provenance_source']=json.loads(model_meta.read_text()).get('source','UNKNOWN')
        except (OSError,ValueError):
            report['model_provenance_source']='UNKNOWN'
    folder=Path(args.out);folder.mkdir(parents=True,exist_ok=True)
    (folder/'results.json').write_text(json.dumps(report,indent=2))
    predictions.to_csv(folder/'out_of_sample_predictions.csv',index=False)
    # Export a separate historical evaluation JSON for the website; never touch live.json or trained models.
    site=Path('web/data')
    if site.is_dir():
        (site/'walk_forward_evaluation.json').write_text(json.dumps(report,indent=2))
    print('\n--- MONTHLY WALK-FORWARD BACKTEST ---')
    print(f"Prepared: {report['prepared_markets']} markets / {report['prepared_events']} events")
    print(f"Evaluated: {report['evaluated_markets']} markets / {report['evaluated_events']} events across {report['n_monthly_folds']} months")
    print(f"Not evaluated in monthly folds: {report['unevaluated_markets']} markets")
    print('\nMONTH         TRAIN MKTS    TEST MKTS    TEST EVENTS')
    for x in report['folds']:
        print(f"{x['window']:<14}{x['train_markets']:<14}{x['test_markets']:<13}{x['test_events']}")
    print('\nMODEL                          BRIER     AUC      LOG LOSS')
    for name,stat in report['metrics'].items():
        auc=stat['auc']
        print(f"{name:<30} {stat['brier']:.4f}   {auc if auc is not None else 'N/A':<8} {stat['log_loss']:.4f}")
    print('\nDelta Brier (with minus without):',report['brier_with_minus_without'])
    print('Delta AUC (with minus without):',report['auc_with_minus_without'])
    if report['bootstrap']:
        print('Event-bootstrap 95% CI for delta Brier:',report['bootstrap']['brier_with_minus_without_ci95'])
        print('Event-bootstrap 95% CI for delta AUC:',report['bootstrap']['auc_with_minus_without_ci95'])
    print('\nSaved:',folder/'results.json','and',folder/'out_of_sample_predictions.csv')
    print('Live models and website unchanged.')


if __name__=='__main__':
    main()
