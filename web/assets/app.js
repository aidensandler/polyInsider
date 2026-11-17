'use strict';
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(x,n=1)=>Number(x).toLocaleString('en-US',{maximumFractionDigits:n,minimumFractionDigits:n});
const num=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:0});
const pct=x=>fmt(Number(x)*100,1)+'%';
const nav=[['index.html','Overview'],['findings.html','Prior findings'],['empirical.html','Real results'],['tracker.html','Live tracker'],['timing.html','Trader timing'],['alerts.html','Alerts'],['methodology.html','Method']];
function layout(page){
 $('ticker').innerHTML='<b>ECE 473 / Princeton</b><span>Research prototype</span><span>Independent forecasts · saved before resolution</span>';
 $('hdr').innerHTML='<div class="inner"><a class="brand" href="index.html">poly<span>insiders</span><span style="font-size:8px;vertical-align:top;letter-spacing:0;margin-left:5px">LAB</span></a><nav>'+nav.map(([href,name])=>'<a '+(href===page?'class="active" ':'')+'href="'+href+'">'+name+'</a>').join('')+'</nav></div>';
 $('ftr').innerHTML='<div class="footer"><div class="inner footcols"><div><strong>PolyInsiders · ECE 473</strong><div>Aiden Sandler &amp; Emilly Antero · November 2026</div></div><div>Research only. Public wallet performance does not prove private information.<div><a href="methodology.html">Methods &amp; limitations</a></div></div></div></div>';
}
async function getJSON(p){const response=await fetch(p+'?v='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error(response.status+' from '+p);return await response.json()}
function setText(id,text){const el=$(id);if(el)el.textContent=text}
function priorCharts(F){
 const a=$('rank-chart');if(a)a.innerHTML=barPairs([{l:'Events',v:.24,alt:.07},{l:'Placebo',v:.07,alt:.07}],.32,'Correlation','').replace(/<div class="bar alt"[^>]*><b>[^<]+<\/b><\/div>/g,'');
 const b=$('auc-chart');if(b)b.innerHTML=barPairs([{l:'Events',v:.65,alt:.5},{l:'Placebo',v:.55,alt:.5}],.78,'AUC','Baseline 0.50');
}
function barPairs(data,max,label,extra){return '<div class="bars">'+data.map(d=>'<div class="item"><div class="barpair"><div class="bar" style="height:'+(d.v/max*100)+'%"><b>'+fmt(d.v,2)+'</b></div><div class="bar alt" style="height:'+(d.alt/max*100)+'%"><b>'+fmt(d.alt,2)+'</b></div></div><div class="chartlabel">'+esc(d.l)+'</div></div>').join('')+'</div><p class="small">'+esc(label)+(extra?' · '+esc(extra):'')+'</p>'}
function renderHistory(){getJSON('data/prior_findings.json').then(F=>{
 setText('reported-at',F.reported_at);
 const d=F.key_numbers; if($('hist-kpis'))$('hist-kpis').innerHTML=d.map(x=>'<div class="metric"><div class="tag">Team-reported · historical</div><div class="val">'+esc(x.value)+'</div><div class="lbl">'+esc(x.label)+'</div></div>').join('');
 if($('history-summary'))$('history-summary').textContent=F.bottom_line;
 if($('rank-chart'))$('rank-chart').innerHTML=barPairs([{l:'Scheduled events',v:.24,alt:.07}],.30,'Earlier vs later rank correlation · green: events · coral: crypto placebo','');
 if($('auc-chart'))$('auc-chart').innerHTML=barPairs([{l:'Predictive ranking',v:.65,alt:.55}],.80,'AUC · green: scheduled events · coral: crypto placebo','');
 if($('result-table'))$('result-table').innerHTML='<div class="tw"><table><thead><tr><th>Reported measure</th><th>Events</th><th>Crypto placebo</th><th>Interpretation</th></tr></thead><tbody>'+F.metrics.map(r=>'<tr><td><strong>'+esc(r.name)+'</strong></td><td>'+esc(r.events)+'</td><td>'+esc(r.placebo)+'</td><td style="text-align:left">'+esc(r.interpretation)+'</td></tr>').join('')+'</tbody></table></div>';
 if($('scope-notes'))$('scope-notes').innerHTML=F.caveats.map(x=>'<p>'+esc(x)+'</p>').join('');
 }).catch(e=>{if($('hist-kpis'))setText('hist-kpis','Cannot load reported findings: '+e.message)})}
function isDemoMode(v){return /DEMO|SYNTHETIC|NOT VALIDATED/i.test(String(v||''))}
function markupMarket(m,i){const pre=Number(m.predicted_yes);const p=Number(m.market_yes_price);const delta=Number(m.prediction_vs_market_pp);const mode=isDemoMode(m.mode)||isDemoMode(m.model_source)?'demo':'real';return '<div class="market" id="market-'+i+'"><div><span class="pill '+(mode==='demo'?'demo':'')+'">'+(mode==='demo'?'Synthetic example':'Public live API')+'</span><h3>'+esc(m.question)+'</h3><div class="meta">Snapshot '+esc(new Date(m.captured_at).toLocaleString())+' · '+num(m.n_taker_trade_rows)+' observed taker fills</div><div class="progress"><div style="width:'+(Math.max(0,Math.min(1,pre))*100)+'%"></div></div><div class="meta">Market price: '+pct(p)+' · Model: '+pct(pre)+'</div></div><div class="prob"><small>Predicted YES</small>'+pct(pre)+'<span class="change '+(delta<0?'neg':'')+'">'+(delta>0?'+':'')+fmt(delta,1)+' pp</span></div></div>'}
function displayLive(F){const demo=isDemoMode(F.mode)||isDemoMode(F.model_source);const age=Math.max(0,(Date.now()-Date.parse(F.updated_at))/60000);const stale=age>10;$('live-status').innerHTML='<span class="pill '+(demo?'demo':'')+'">'+esc(F.mode)+'</span><span>Last captured: '+esc(new Date(F.updated_at).toLocaleString())+'</span><span>'+num(F.markets.length)+' markets</span>';
 if($('latest-ticker'))$('latest-ticker').textContent=demo?'Synthetic sample — not a live signal':'Latest saved live API snapshot';
 $('market-feed').innerHTML=F.markets.length?F.markets.map(markupMarket).join(''):'<div class="slot">No valid open market forecasts saved yet. Check the collector status.</div>';
 if($('live-explanation'))setText('live-explanation',F.note);
 if($('wallet-feed')){const uniq=new Map();F.markets.forEach(m=>(m.top_wallets||[]).forEach(w=>{if(!uniq.has(w.wallet))uniq.set(w.wallet,w)}));const ws=[...uniq.values()].sort((a,b)=>b.prior_score-a.prior_score).slice(0,8);$('wallet-feed').innerHTML=ws.map(w=>'<div class="wallet-row"><span><strong>'+esc(w.wallet)+'</strong><div class="small">'+num(w.prior_events)+' previously resolved events · '+esc(w.direction)+' observed flow</div></span><span class="nobr">'+(w.prob_future_outperform==null?'Insufficient prior events':'Predicted future edge '+pct(w.prob_future_outperform))+'</span></div>').join('')||'<div class="small">No wallet records in last snapshot.</div>'}
 if($('model-provenance'))$('model-provenance').textContent='Market model: '+String(F.model_source||'unknown')+' · Wallet model: '+String(F.wallet_model_source||'none')+' · Snapshot captured '+String(F.updated_at||'unknown');
 if($('errors'))$('errors').textContent=F.errors?.length?F.errors.length+' API market requests failed; those markets were not scored.':'No captured market errors in this snapshot.';
}
async function loadLive(){try{let F=await getJSON('data/live.json');displayLive(F)}catch(e){setText('market-feed','No live data file: '+e.message)}}
async function renderBacktest(){try{const W=await getJSON('data/wallet_backtest_demo.json');setText('wallet-test','Pipeline test on synthetic data (not our Polymarket results): '+num(W.test_wallets)+' wallet-window examples, AUC '+fmt(W.test_auc,3)+', Brier '+fmt(W.test_brier,3)+' vs constant-rate '+fmt(W.base_rate_brier,3)+'. Not real-data evidence.')}catch(e){setText('wallet-test','No wallet backtest file: '+e.message)}try{const F=await getJSON('data/backtest_demo.json');setText('demo-test','Pipeline test on synthetic data only: '+F.test_markets+' held-out markets; model Brier '+fmt(F.model.brier,3)+' vs market '+fmt(F.market_baseline.brier,3)+'. This is NOT a result from the team’s real Polymarket archive.')}catch(e){setText('demo-test','No synthetic backtest available: '+e.message)}}
