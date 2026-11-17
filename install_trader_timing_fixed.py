from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import duckdb

root = Path.home() / "Desktop/PolyInsiders_ECE473_Updated"
source = root / "data_events_full/live"
v4 = root / "results_v4"
web = root / "web"

required = [
    source / "fills_clean.parquet",
    source / "markets_coded.parquet",
    source / "wallet_codes.parquet",
    v4 / "likely_informed_wallets.csv",
    v4 / "validation_report.json",
    web / "assets/app.js",
]
for p in required:
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

backup = root / ("website_backup_timing_" +
                 datetime.now().strftime("%Y%m%d_%H%M%S"))
shutil.copytree(web, backup)
print("Website backed up:", backup, flush=True)

report = json.loads((v4 / "validation_report.json").read_text())
split = report["events"]["split"]
cutoff = int(datetime.fromisoformat(split).replace(
    tzinfo=timezone.utc).timestamp())

def quote(p):
    return str(p).replace("'", "''")

fills = quote(source / "fills_clean.parquet")
markets = quote(source / "markets_coded.parquet")
wallets = quote(source / "wallet_codes.parquet")
flagfile = quote(v4 / "likely_informed_wallets.csv")

con = duckdb.connect()
con.execute("SET TimeZone='UTC'")
con.execute("SET memory_limit='4GB'")

print("Matching flagged wallets...", flush=True)
con.execute(f"""
CREATE TEMP TABLE flagged AS
SELECT DISTINCT w.code
FROM read_parquet('{wallets}') w
JOIN read_csv_auto('{flagfile}') f
 ON lower(w.wallet)=lower(f.wallet)
WHERE TRY_CAST(f.informed AS BOOLEAN)=TRUE
""")

nflag = con.execute("SELECT COUNT(*) FROM flagged").fetchone()[0]
if nflag == 0:
    raise SystemExit("No flagged wallets matched the archive")

print("Selecting post-cutoff markets...", flush=True)
con.execute(f"""
CREATE TEMP TABLE matched AS
SELECT DISTINCT f.condition_id
FROM read_parquet('{fills}') f
JOIN flagged x ON f.wallet=x.code
JOIN read_parquet('{markets}') m
 ON f.condition_id=m.condition_id
WHERE f.timestamp >= {cutoff}
 AND m.info_time IS NOT NULL
 AND f.timestamp < epoch(CAST(m.info_time AS TIMESTAMPTZ))
""")

nmarkets = con.execute(
    "SELECT COUNT(*) FROM matched"
).fetchone()[0]

if nmarkets == 0:
    raise SystemExit("No qualifying post-cutoff markets found")

print("Calculating timing distributions...", flush=True)
rows = con.execute(f"""
WITH trades AS (
 SELECT
  CASE WHEN x.code IS NULL THEN 'other'
       ELSE 'flagged' END AS grp,
  COALESCE(m.info_time_source,'unknown') AS src,
  CAST(EXTRACT(hour FROM to_timestamp(f.timestamp)) AS INT) AS hr,
  CAST(EXTRACT(dow FROM to_timestamp(f.timestamp)) AS INT) AS dow,
  (epoch(CAST(m.info_time AS TIMESTAMPTZ))
      - f.timestamp)/3600.0 AS lead_hours,
  CAST(f.usdc AS DOUBLE) AS usd
 FROM read_parquet('{fills}') f
 JOIN matched mm ON f.condition_id=mm.condition_id
 JOIN read_parquet('{markets}') m
  ON f.condition_id=m.condition_id
 LEFT JOIN flagged x ON f.wallet=x.code
 WHERE f.timestamp >= {cutoff}
  AND m.info_time IS NOT NULL
  AND f.timestamp < epoch(CAST(m.info_time AS TIMESTAMPTZ))
),
bucketed AS (
 SELECT grp,src,hr,dow,usd,
 CASE
  WHEN lead_hours < 1 THEN '0-1h'
  WHEN lead_hours < 6 THEN '1-6h'
  WHEN lead_hours < 12 THEN '6-12h'
  WHEN lead_hours < 24 THEN '12-24h'
  WHEN lead_hours < 48 THEN '24-48h'
  ELSE '48h+'
 END AS lead
 FROM trades WHERE usd > 0
)
SELECT grp,src,hr,dow,lead,
 COUNT(*) AS fills,SUM(usd) AS dollars
FROM bucketed
GROUP BY 1,2,3,4,5
""").fetchall()

if not rows:
    raise SystemExit("No qualifying trade records")

data = {
    "split_date": split,
    "flagged_wallets": nflag,
    "matched_markets": nmarkets,
    "lead_buckets": [
        "0-1h","1-6h","6-12h",
        "12-24h","24-48h","48h+"
    ],
    "rows": [
        {
            "group": g, "source": s,
            "hour": h, "dow": d, "lead": l,
            "fills": n, "dollars": round(float(v), 2)
        }
        for g,s,h,d,l,n,v in rows
    ],
    "caveat": "Retrospective wallet classification and approximate information times."
}

(web / "data").mkdir(exist_ok=True)
(web / "data/trader_timing.json").write_text(json.dumps(data))

page = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trader Timing | PolyInsiders</title>
<link rel="stylesheet" href="assets/style.css">
<style>
body{background:#0d1725;color:#e9f3ff;font-family:system-ui,Arial,sans-serif;margin:0}
header{padding:22px 5%;background:#142235;border-bottom:1px solid #34465e}
header a{color:#c7d9eb;text-decoration:none;margin-right:15px;font-size:14px}
header strong{color:#6ce5d0;font-size:23px;margin-right:24px}
main{max-width:1100px;margin:auto;padding:35px 22px 80px}
h1{font-size:42px;margin-bottom:10px}
p{color:#bacadd;line-height:1.7}
.panel{background:#17263a;border:1px solid #354a61;border-radius:14px;padding:25px;margin-top:18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}
.stat{background:#17263a;border:1px solid #354a61;border-radius:12px;padding:22px}
.stat b{display:block;font-size:27px}
.stat small{color:#a7bcd3}
select{background:#23364e;color:white;border:1px solid #52708b;padding:10px;border-radius:7px;margin:8px 12px 8px 0}
#plot{display:flex;gap:5px;align-items:end;height:265px;min-width:730px}
.group{flex:1;height:100%;display:flex;flex-direction:column;justify-content:end;align-items:center}
.pair{height:225px;width:100%;display:flex;gap:3px;align-items:end;justify-content:center}
.bar{width:38%;border-radius:3px 3px 0 0;min-height:1px}
.flag{background:#6ce5d0}
.other{background:#8fa8ff}
.tick{font-size:10px;color:#b9cee3;margin-top:7px}
.legend{margin:12px 0;color:#cbd8e9}
.dot{display:inline-block;width:11px;height:11px;margin:0 6px 0 14px}
.note{border-left:3px solid #d4ab69;padding:15px;background:#302b25}
</style>
</head>
<body>
<header>
<strong>polyinsiders LAB</strong>
<a href="index.html">Overview</a>
<a href="findings.html">Prior findings</a>
<a href="empirical.html">Real results</a>
<a href="tracker.html">Live tracker</a>
<a href="timing.html">Trader timing</a>
<a href="methodology.html">Method</a>
</header>
<main>
<p>ECE 473 · PRINCETON · RESEARCH PROTOTYPE</p>
<h1>When do flagged traders trade?</h1>
<p>Compare historically flagged wallets with other participants in the same markets, using trade timestamps from the full Polymarket archive.</p>
<div class="stats">
<div class="stat"><b id="wallets">—</b><small>Flagged wallets matched</small></div>
<div class="stat"><b id="markets">—</b><small>Markets analyzed</small></div>
<div class="stat"><b id="cutoff">—</b><small>Analysis cutoff (UTC)</small></div>
</div>
<section class="panel">
<h2>Historical trading patterns</h2>
<select id="axis">
<option value="hour">Hour of day (UTC)</option>
<option value="lead">Time before information</option>
<option value="dow">Day of week</option>
</select>
<select id="metric">
<option value="fills">Fill frequency</option>
<option value="dollars">Recorded fill-side USD</option>
</select>
<select id="source"><option value="all">All information-time sources</option></select>
<div class="legend">
<span class="dot flag"></span>Flagged wallets
<span class="dot other"></span>Other wallets
</div>
<div style="overflow-x:auto"><div id="plot"></div></div>
<p id="summary"></p>
</section>
<section class="panel">
<h2>How to interpret this chart</h2>
<p>Every bar shows the percentage of activity within its own wallet group. This normalizes for the very different number of flagged and other traders.</p>
<p class="note">Flagged wallets are not proven insiders. Many information times are scheduled-event proxies rather than independently verified first announcements. Results are observational and may reflect market selection, liquidity, or trading schedules. Some transaction sides may be recorded separately.</p>
</section>
</main>
<script>
let DATA;
const el=id=>document.getElementById(id);
const fmt=x=>Number(x).toLocaleString('en-US',{maximumFractionDigits:1});
const days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];

function draw(){
 const axis=el('axis').value;
 const metric=el('metric').value;
 const source=el('source').value;
 const labels=axis==='hour'
   ?Array.from({length:24},(_,i)=>String(i).padStart(2,'0'))
   :axis==='dow'?days:DATA.lead_buckets;
 const vals={flagged:labels.map(()=>0),other:labels.map(()=>0)};
 for(const r of DATA.rows){
   if(source!=='all'&&r.source!==source)continue;
   const i=axis==='hour'?r.hour:axis==='dow'?r.dow:labels.indexOf(r.lead);
   if(i>=0&&i<labels.length)vals[r.group][i]+=r[metric];
 }
 const sums={};
 const pct={};
 for(const g of ['flagged','other']){
   sums[g]=vals[g].reduce((a,b)=>a+b,0);
   pct[g]=vals[g].map(v=>sums[g]?v/sums[g]*100:0);
 }
 const max=Math.max(0.01,...pct.flagged,...pct.other)*1.1;
 el('plot').innerHTML=labels.map((label,i)=>`
 <div class="group">
 <div class="pair">
 <div class="bar flag" style="height:${100*pct.flagged[i]/max}%"
 title="Flagged: ${pct.flagged[i].toFixed(2)}%"></div>
 <div class="bar other" style="height:${100*pct.other[i]/max}%"
 title="Other: ${pct.other[i].toFixed(2)}%"></div>
 </div><span class="tick">${label}</span></div>`).join('');
 el('summary').textContent=
 `Flagged activity: ${fmt(sums.flagged)}; other activity: ${fmt(sums.other)}. Each group sums to 100%.`;
}
async function start(){
 try{
  const response=await fetch('data/trader_timing.json',{cache:'no-store'});
  if(!response.ok)throw Error('Data file missing');
  DATA=await response.json();
  el('wallets').textContent=fmt(DATA.flagged_wallets);
  el('markets').textContent=fmt(DATA.matched_markets);
  el('cutoff').textContent=DATA.split_date;
  const srcs=[...new Set(DATA.rows.map(r=>r.source))].sort();
  el('source').innerHTML='<option value="all">All information-time sources</option>'+
    srcs.map(s=>`<option value="${s}">${s.replace(/_/g,' ')}</option>`).join('');
  for(const id of ['axis','metric','source'])
    el(id).addEventListener('change',draw);
  draw();
 }catch(e){
  el('summary').textContent='Could not load analysis: '+e.message;
 }
}
start();
</script>
</body>
</html>"""

(web / "timing.html").write_text(page)

navfile = web / "assets/app.js"
nav = navfile.read_text()
item = "['timing.html','Trader timing']"
if item not in nav:
    old = "['tracker.html','Live tracker']"
    if old in nav:
        navfile.write_text(nav.replace(old, old + "," + item, 1))
    else:
        print("Navigation format differs; page has its own links.")

print()
print("SUCCESS: Trader Timing installed!")
print("Flagged wallets matched:", nflag)
print("Markets analyzed:", nmarkets)
print("Rows aggregated:", len(rows))
print("Open: http://localhost:8000/timing.html")
