"""Clearly labeled synthetic example for UI and pipeline testing. NOT empirical evidence."""
from __future__ import annotations
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .core import write_training
from .live import score_market,append_predictions
from .wallet_model import train_wallet_file

def make_history(path,seed=473,n=220):
    rng=np.random.default_rng(seed);start=datetime(2025,1,10,tzinfo=timezone.utc)
    wallets=[f'demo_wallet_{i:03d}' for i in range(70)]; skilled=set(wallets[:15]); rows=[]
    for j in range(n):
        resolution=start+timedelta(days=j*2)
        snap=resolution-timedelta(days=1)
        latent=float(rng.beta(2.8,2.8))
        p=float(np.clip(latent+rng.normal(0,.12),.06,.94))
        y=int(rng.random()<latent)
        selected=rng.choice(wallets,size=int(rng.integers(13,31)),replace=False)
        for wallet in selected:
            if wallet in skilled:
                direction=y if rng.random()<.8 else 1-y
            else:
                direction=int(rng.random()<p)
            rows.append({'event_id':f'demo_event_{j:04d}',
               'condition_id':f'demo_condition_{j:04d}',
               'title':f'Synthetic binary event #{j+1}',
               'snapshot_time':snap.isoformat(),'resolution_time':resolution.isoformat(),
               'yes_price':round(p,4),'wallet':wallet,'direction':'YES' if direction else 'NO',
               'amount_usd':round(float(np.exp(rng.normal(3,1.0))),2),
               'resolved_yes':y})
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False)
    return len(rows)

def build_demo(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    hist=root/'data'/'synthetic_history.csv'
    n=make_history(hist)
    report=write_training(hist,root/'data'/'run',source='SYNTHETIC DEMONSTRATION')
    wallet_report=train_wallet_file(hist,root/'data'/'run',source='SYNTHETIC DEMONSTRATION')
    model=json.loads((root/'data'/'run'/'model.json').read_text())
    state=json.loads((root/'data'/'run'/'wallet_state.json').read_text())
    wallet_predictors={r['wallet']:r for r in pd.read_csv(root/'data'/'run'/'wallet_probabilities.csv').to_dict('records')}
    rng=np.random.default_rng(763)
    rng_wallets=list(state);demo_predictions=[];timestamp=datetime.now(timezone.utc).isoformat()
    for j in range(7):
        price=float(rng.uniform(.16,.84));trades=[]
        for wi in rng.choice(rng_wallets,16,replace=False):
            direction='YES' if rng.random() < price else 'NO'
            trades.append({'wallet':wi,'direction':direction,'amount_usd':float(rng.uniform(10,290))})
        m={'event_id':f'open_demo_{j}','condition_id':f'open_demo_{j}',
           'question':['Will the proposed economic measure pass?','Will inflation fall below the stated threshold?',
             'Will the next policy meeting change rates?','Will the election result favor the incumbent?',
             'Will the scheduled jobs report exceed consensus?', 'Will the initiative pass before December?',
             'Will the next scheduled announcement be positive?'][j],
           'yes_price':price,'volume_usd':round(float(rng.uniform(45000,940000)),0),
           'end_date':(datetime.now(timezone.utc)+timedelta(days=j+2)).isoformat(), 'slug':''}
        demo_predictions.append(score_market(m,trades,model,state,captured_at=timestamp,
                   mode='SYNTHETIC DEMO — UNRESOLVED SIMULATION',
                   wallet_probability_records=wallet_predictors))
    web=root/'web';(web/'data').mkdir(parents=True,exist_ok=True)
    (web/'data'/'live.json').write_text(json.dumps({'updated_at':timestamp,
         'mode':'SYNTHETIC DEMO — NOT LIVE MARKETS',
         'model_source':'SYNTHETIC DEMONSTRATION','wallet_model_source':'SYNTHETIC DEMONSTRATION',
         'markets':demo_predictions,'errors':[],
         'note':'All seven open example markets and wallet addresses are synthetic. No real outcomes or live API calls.'},indent=2))
    (web/'data'/'backtest_demo.json').write_text(json.dumps({'source':'SYNTHETIC DEMONSTRATION',**report},indent=2))
    append_predictions(demo_predictions,root/'data'/'run'/'demo_prediction_log.csv')
    (web/'data'/'wallet_backtest_demo.json').write_text(json.dumps({'source':'SYNTHETIC DEMONSTRATION',**wallet_report},indent=2))
    return {'generated_rows':n,'market_report':report,'wallet_report':wallet_report,'demo_markets':len(demo_predictions)}
