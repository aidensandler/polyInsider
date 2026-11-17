"""Temporal wallet prediction, provenance and durable logging tests."""
import json
from datetime import datetime,timezone,timedelta
import pandas as pd
import pytest
from polyinsider.demo import make_history
from polyinsider.wallet_model import (event_wallet_observations,prior_wallet_features,
    training_window,train_wallet_outperformance,train_wallet_file,prob_wallet)
from polyinsider.live import append_audit_chain,verify_audit_chain,score_market,fetch_taker_trades


def test_trained_wallet_model_really_uses_future_holdout(tmp_path):
    filename=tmp_path/'synthetic.csv';make_history(filename,n=220)
    model,report,state=train_wallet_outperformance(pd.read_csv(filename),source='SYNTHETIC DEMONSTRATION')
    assert report['train_labels_end_before']==report['test_features_frozen_at']
    assert report['test_wallets']>0 and 0<=report['test_auc']<=1
    assert report['test_brier']>=0
    assert 'NOT private information' in report['target']
    assert len(state)>0
    assert 0<prob_wallet(model,state.iloc[0])<1


def test_wallet_features_do_not_use_unresolved_sibling_outcomes():
    # event 1 has two markets, the late one extends the entire event end.
    entries=[{'event_id':'e1','condition_id':'m1','title':'m1','snapshot_time':'2026-01-01T00:00Z',
      'resolution_time':'2026-01-02T00:00Z','yes_price':.5,'wallet':'w','direction':'YES',
      'amount_usd':20,'resolved_yes':1},
      {'event_id':'e1','condition_id':'m2','title':'m2','snapshot_time':'2026-01-01T00:00Z',
      'resolution_time':'2026-01-10T00:00Z','yes_price':.6,'wallet':'w','direction':'NO',
      'amount_usd':20,'resolved_yes':0},
      {'event_id':'e2','condition_id':'m3','title':'m3','snapshot_time':'2026-01-04T00:00Z',
      'resolution_time':'2026-01-15T00:00Z','yes_price':.3,'wallet':'w','direction':'YES',
      'amount_usd':20,'resolved_yes':0}]
    rows=event_wallet_observations(pd.DataFrame(entries))
    assert prior_wallet_features(rows,'2026-01-05T00:00Z',min_prior=1).empty
    available=prior_wallet_features(rows,'2026-01-11T00:00Z',min_prior=1)
    assert available.iloc[0].prior_events==1
    assert abs(available.iloc[0].prior_mean_edge_raw-.55)<1e-7


def test_wallet_targets_are_future_only(tmp_path):
    fn=tmp_path/'synthetic.csv';make_history(fn,n=150)
    rows=event_wallet_observations(pd.read_csv(fn))
    cutoff='2025-05-10T00:00Z'; end='2025-07-10T00:00Z'
    a=training_window(rows,cutoff,end)
    assert len(a)>0
    # Alter future result edges: earlier wallet features at cutoff stay unchanged.
    later=rows.copy();later.loc[later.event_end>=pd.Timestamp(cutoff),'edge']*=-1
    b=training_window(later,cutoff,end)
    joined=a.merge(b,on='wallet',suffixes=('_a','_b'))
    assert (joined.prior_mean_edge_a==joined.prior_mean_edge_b).all()
    assert (joined.future_mean_edge_a==-joined.future_mean_edge_b).all()


def test_frozen_wallet_probabilities_attached_to_open_market():
    base={'event_id':'a','condition_id':'m','question':'q','yes_price':.4}
    model={'intercept':0.0,'coefficients':{k:0 for k in ['logit_price','net_flow','skill_flow','wallet_count','log_volume']},'source':'SYNTHETIC DEMONSTRATION'}
    p=score_market(base,[{'wallet':'w','direction':'YES','amount_usd':10}],model,{'w':[15,3]},
        wallet_probability_records={'w':{'prob_future_outperform':.735,'source':'SYNTHETIC DEMONSTRATION'}})
    assert p['top_wallets'][0]['prob_future_outperform']==.735
    assert p['predicted_yes']==.5


def test_audit_chain_detects_edit(tmp_path):
    f=tmp_path/'audit.jsonl'
    append_audit_chain([{'condition_id':'m1','predicted_yes':.42},{'condition_id':'m2','predicted_yes':.72}],f)
    assert verify_audit_chain(f)['verified_rows']==2
    with f.open('a') as fd:fd.write('')
    tampered=f.read_text().replace('.42','.43',1)
    f.write_text(tampered)
    with pytest.raises(ValueError,match='corrupted'):verify_audit_chain(f)


def test_trade_collector_excludes_future_and_stale_data():
    now=datetime(2026,11,10,12,tzinfo=timezone.utc)
    class FakeResp:
        def raise_for_status(self):pass
        def json(self):return {'data':[
            {'proxy_wallet':'good','outcome':'Yes','side':'BUY','price':.5,'size':10,'timestamp':int((now-timedelta(hours=4)).timestamp())},
            {'proxy_wallet':'future','outcome':'Yes','side':'BUY','price':.5,'size':10,'timestamp':int((now+timedelta(hours=1)).timestamp())},
            {'proxy_wallet':'stale','outcome':'No','side':'BUY','price':.5,'size':10,'timestamp':int((now-timedelta(hours=60)).timestamp())},
            {'proxy_wallet':'missing','outcome':'No','side':'BUY','price':.5,'size':10}
            ],'pagination':{'next_cursor':None}}
    class FakeSession:
        def get(self,*args,**kwargs):return FakeResp()
    t=fetch_taker_trades(FakeSession(),'0xabcdef',asof=now)
    assert [x['wallet'] for x in t]==['good']

def test_streaming_import_agrees_with_small_import(tmp_path):
    from polyinsider.prepare import prepare_snapshots,prepare_snapshots_streaming
    rows=[{'condition_id':'ab','proxy_wallet':'w','outcome':'Yes','side':'BUY','price':.54,
           'size':10,'timestamp':'2026-01-01T09:00Z'},
          {'condition_id':'ab','proxy_wallet':'w','outcome':'No','side':'SELL','price':.47,
           'size':5,'timestamp':'2026-01-01T10:00Z'},
          {'condition_id':'ab','proxy_wallet':'v','outcome':'No','side':'BUY','price':.48,
           'size':10,'timestamp':'2026-01-01T11:00Z'}]
    markets=[{'condition_id':'ab','event_id':'e','title':'Q',
        'snapshot_time':'2026-01-01T12:00Z','resolution_time':'2026-01-02T12:00Z','resolved_yes':1}]
    fills=tmp_path/'fills.csv';metadata=tmp_path/'metadata.csv'
    pd.DataFrame(rows).to_csv(fills,index=False);pd.DataFrame(markets).to_csv(metadata,index=False)
    a=tmp_path/'std.csv';b=tmp_path/'stream.csv'
    prepare_snapshots(fills,metadata,a)
    prepare_snapshots_streaming(fills,metadata,b,chunksize=1)
    sa=pd.read_csv(a).sort_values('wallet').reset_index(drop=True)
    sb=pd.read_csv(b).sort_values('wallet').reset_index(drop=True)
    pd.testing.assert_frame_equal(sa,sb,check_exact=False)

def test_capture_live_mock_end_to_end_writes_pre_resolution_forecasts(tmp_path):
    from polyinsider.live import capture_live
    from polyinsider.demo import make_history
    from polyinsider.core import write_training
    import json
    from datetime import datetime,timezone
    hist=tmp_path/'synthetic.csv';make_history(hist,n=100)
    write_training(hist,tmp_path/'run',source='SYNTHETIC DEMONSTRATION')
    now=int(datetime.now(timezone.utc).timestamp())-60
    class Resp:
        def __init__(self,p):self.p=p
        def raise_for_status(self):pass
        def json(self):return self.p
    class FakeSession:
        def get(self,url,params=None,timeout=None):
            if url.endswith('/markets'):
                return Resp([{'active':True,'closed':False,'id':55,'conditionId':'0xabcde',
                    'question':'Fictional open event','outcomes':'["Yes","No"]',
                    'outcomePrices':'["0.4","0.6"]','volumeNum':100000}])
            if url.endswith('/v2/trades'):
                return Resp({'data':[{'proxy_wallet':'demo_wallet_%03d'%i,'outcome':'Yes',
                   'side':'BUY','price':.4,'size':10,'timestamp':now} for i in range(4)],
                   'pagination':{'next_cursor':None}})
            raise RuntimeError(url)
    result=capture_live(tmp_path/'run'/'model.json',tmp_path/'web',limit=1,
        session=FakeSession(),allow_demo_model=True)
    assert len(result['markets'])==1
    assert 'SYNTHETIC' in result['mode']
    assert (tmp_path/'run'/'predictions.csv').exists()
    assert verify_audit_chain(tmp_path/'run'/'predictions_audit.jsonl')['verified_rows']==1
    saved=json.loads((tmp_path/'web'/'data'/'live.json').read_text())
    assert saved['markets'][0]['condition_id']=='0xabcde'
