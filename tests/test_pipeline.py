from pathlib import Path
import json
import pandas as pd
import pytest
from polyinsider.demo import make_history
from polyinsider.core import normalize_history,chronological_features,wallet_state_asof,market_features,train_evaluate
from polyinsider.live import fetch_open,fetch_taker_trades,score_market,append_predictions,evaluate_saved_predictions,capture_live

class FakeResp:
    def __init__(self,data):self.data=data
    def raise_for_status(self):pass
    def json(self):return self.data
class FakeSession:
    def get(self,url,params=None,timeout=None):
        if '/markets' in url:return FakeResp([{'active':True,'closed':False,'id':5,'conditionId':'0xdef',
           'question':'Synthetic example question','outcomes':'["Yes","No"]',
           'outcomePrices':'["0.4","0.6"]','volumeNum':200000}])
        if '/v2/trades' in url:return FakeResp({'data':[{'proxy_wallet':'0x123','outcome':'Yes',
            'side':'BUY','price':.4,'size':10,'timestamp':1}],
            'pagination':{'next_cursor':None}})
        raise AssertionError(url)

def test_demo_history_train(tmp_path):
    h=tmp_path/'x.csv';make_history(h,n=110)
    d=normalize_history(pd.read_csv(h))
    f=chronological_features(d)
    assert len(f)==110
    model,state,rep,_=train_evaluate(d,source='TEST SYNTHETIC')
    assert rep['trained_markets']+rep['test_markets']<=110
    assert rep['train_last_resolution']<rep['test_first_snapshot']
    assert 0<rep['model']['brier']<1
    assert 'logit_price' in model['coefficients']

def test_wallets_not_learn_same_event_or_future():
    # At a snapshot before any event resolves, the wallet feature must remain zero.
    d=pd.DataFrame([
     dict(event_id='event1',condition_id='m1',title='m1',snapshot_time='2025-01-01T00:00:00Z',
          resolution_time='2025-01-10T00:00:00Z',yes_price=.5,wallet='a',direction='YES',amount_usd=20,resolved_yes=1),
     dict(event_id='event1',condition_id='m2',title='m2',snapshot_time='2025-01-04T00:00:00Z',
          resolution_time='2025-01-06T00:00:00Z',yes_price=.4,wallet='a',direction='YES',amount_usd=20,resolved_yes=1),
     dict(event_id='event2',condition_id='m3',title='m3',snapshot_time='2025-01-07T00:00:00Z',
          resolution_time='2025-01-15T00:00:00Z',yes_price=.6,wallet='a',direction='YES',amount_usd=20,resolved_yes=1),
     dict(event_id='event3',condition_id='m4',title='m4',snapshot_time='2025-01-11T00:00:00Z',
          resolution_time='2025-01-17T00:00:00Z',yes_price=.6,wallet='a',direction='YES',amount_usd=20,resolved_yes=1)])
    features=chronological_features(d)
    assert features[features.condition_id=='m3'].skill_flow.iloc[0]==0
    # Grouped event1 only becomes eligible on 1/10, not 1/06.
    state=wallet_state_asof(d,'2025-01-08T00:00:00Z')
    assert state=={}
    assert wallet_state_asof(d,'2025-01-11T00:00:00Z')['a'][0]==1

def test_reject_post_resolution_data():
    row=dict(event_id='a',condition_id='a',title='q',snapshot_time='2025-02-02',
         resolution_time='2025-02-01',yes_price=.5,wallet='a',direction='YES',amount_usd=20,resolved_yes=1)
    with pytest.raises(ValueError,match='precede'):normalize_history(pd.DataFrame([row]))

def test_real_api_mock():
    m=fetch_open(FakeSession(),limit=3)
    assert len(m)==1 and m[0]['yes_price']==.4
    from datetime import datetime,timezone
    t=fetch_taker_trades(FakeSession(),'0xdef',asof=datetime(1970,1,1,0,0,2,tzinfo=timezone.utc))
    assert t[0]['amount_usd']==4
    assert t[0]['direction']=='YES'

def test_closed_market_not_scored():
    with pytest.raises(ValueError,match='closed'):
        score_market({'closed':True},{},{},{})

def test_live_evaluation_before_resolution_only(tmp_path):
    records=[{'captured_at':'2026-01-01T12:00:00Z','event_id':'e','condition_id':'m',
      'question':'q','market_yes_price':.4,'predicted_yes':.7,'model_source':'REAL','mode':'LIVE'},
      {'captured_at':'2026-01-03T12:00:00Z','event_id':'e','condition_id':'m',
      'question':'q','market_yes_price':.5,'predicted_yes':.8,'model_source':'REAL','mode':'LIVE'}]
    csv_path=tmp_path/'predictions.csv';append_predictions(records,csv_path)
    truth=tmp_path/'resolved.csv';pd.DataFrame([{'condition_id':'m','resolved_yes':1,
       'resolution_time':'2026-01-02T00:00:00Z'}]).to_csv(truth,index=False)
    report=evaluate_saved_predictions(csv_path,truth,tmp_path/'out.json')
    assert report['n_verified_pre_resolution_markets']==1
    assert report['model']['brier']==.09

def test_synthetic_model_live_guard(tmp_path):
    p=tmp_path/'model.json';p.write_text(json.dumps({'source':'SYNTHETIC DEMONSTRATION'}))
    with pytest.raises(ValueError,match='Synthetic model'):
       capture_live(p,tmp_path/'web',session=FakeSession())


def test_prepare_fills_to_snapshots(tmp_path):
    from polyinsider.prepare import prepare_snapshots
    t=tmp_path/'trades.csv';m=tmp_path/'metadata.csv'
    pd.DataFrame([{'condition_id':'abc','proxy_wallet':'x','outcome':'Yes','side':'BUY',
      'price':.54,'size':10,'timestamp':'2026-01-01T10:00:00Z'},
      {'condition_id':'abc','proxy_wallet':'x','outcome':'No','side':'SELL',
      'price':.47,'size':5,'timestamp':'2026-01-01T11:00:00Z'}]).to_csv(t,index=False)
    pd.DataFrame([{'condition_id':'abc','event_id':'e','title':'question',
      'snapshot_time':'2026-01-01T12:00:00Z','resolution_time':'2026-01-02T00:00:00Z','resolved_yes':1}]).to_csv(m,index=False)
    report=prepare_snapshots(t,m,tmp_path/'prepared.csv')
    d=pd.read_csv(tmp_path/'prepared.csv');assert len(d)==1
    assert d.direction.iloc[0]=='YES'
    assert report['produced_markets']==1
