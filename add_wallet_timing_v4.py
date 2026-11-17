from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import json
import shutil
import duckdb

root = Path.home() / "Desktop/PolyInsiders_ECE473_Updated"
web = root / "web"
data = root / "data_events_full/live"
v4 = root / "results_v4"

required = [
    web/"timing.html",
    web/"index.html",
    data/"fills_clean.parquet",
    data/"markets_coded.parquet",
    data/"wallet_codes.parquet",
    v4/"likely_informed_wallets.csv",
    v4/"validation_report.json"
]
for p in required:
    if not p.is_file():
        raise SystemExit("Missing required file: " + str(p))

report = json.loads((v4/"validation_report.json").read_text())
cal = report["events"]["calibration"]
ev = report["events"]["evaluation"]
start = report["events"]["split"]
cutoff = int(datetime.fromisoformat(start).replace(
    tzinfo=timezone.utc).timestamp())
holdout = int(datetime(2026, 9, 1, tzinfo=timezone.utc).timestamp())

def q(p):
    return str(p).replace("'", "''")

con = duckdb.connect()
con.execute("SET TimeZone='UTC'")
con.execute("SET memory_limit='4GB'")

print("Matching V4 wallet addresses...", flush=True)

con.execute(f"""
CREATE TEMP TABLE flagged AS
SELECT DISTINCT c.code, lower(c.wallet) address
FROM read_parquet('{q(data/"wallet_codes.parquet")}') c
JOIN read_csv_auto('{q(v4/"likely_informed_wallets.csv")}') x
ON lower(c.wallet)=lower(x.wallet)
WHERE TRY_CAST(x.informed AS BOOLEAN)=TRUE
""")

nflags = con.execute(
    "SELECT count(*) FROM flagged"
).fetchone()[0]

if nflags == 0:
    raise SystemExit("No flagged wallets matched")

base = f"""
FROM read_parquet('{q(data/"fills_clean.parquet")}') f
JOIN read_parquet('{q(data/"markets_coded.parquet")}') m
ON f.condition_id=m.condition_id
"""

period = (
    f"CASE WHEN f.timestamp < {holdout} "
    "THEN 'discovery' ELSE 'validation' END"
)

where = (
    f"f.timestamp >= {cutoff} "
    "AND m.info_time IS NOT NULL "
    "AND f.timestamp < epoch(CAST(m.info_time AS TIMESTAMPTZ)) "
    "AND f.usdc > 0"
)

cols = f"""
f.condition_id market,
{period} period,
CAST(EXTRACT(hour FROM
 (to_timestamp(f.timestamp) AT TIME ZONE 'UTC'))
 AS INT) utc_hour,
CAST(EXTRACT(hour FROM
 (to_timestamp(f.timestamp) AT TIME ZONE 'America/New_York'))
 AS INT) ny_hour,
count(*) fills,
sum(CAST(f.usdc AS DOUBLE)) dollars
"""

print("Calculating individual wallet timing...", flush=True)

con.execute(f"""
CREATE TEMP TABLE own AS
SELECT fl.address wallet, {cols}
{base}
JOIN flagged fl ON fl.code=f.wallet
WHERE {where}
GROUP BY 1,2,3,4,5
""")

con.execute("""
CREATE TEMP TABLE relevant AS
SELECT DISTINCT market FROM own
""")

print("Calculating same-market comparisons...", flush=True)

con.execute(f"""
CREATE TEMP TABLE peers AS
SELECT {cols}
{base}
JOIN relevant r ON r.market=f.condition_id
LEFT JOIN flagged fl ON fl.code=f.wallet
WHERE fl.code IS NULL AND {where}
GROUP BY 1,2,3,4
""")

def blank():
    return {
        "utc": [[0.0]*24 for _ in range(2)],
        "ny": [[0.0]*24 for _ in range(2)]
    }

def add(target, utc, ny, fills, dollars):
    for zone, hour in [("utc",utc),("ny",ny)]:
        target[zone][0][hour] += float(fills)
        target[zone][1][hour] += float(dollars)

own = defaultdict(blank)
peers = defaultdict(blank)

for wallet, market, p, utc, ny, fills, usd in con.execute(
    "SELECT * FROM own"
).fetchall():
    for t in (p, "all"):
        add(own[(wallet,market,t)],utc,ny,fills,usd)

for market,p,utc,ny,fills,usd in con.execute(
    "SELECT * FROM peers"
).fetchall():
    for t in (p,"all"):
        add(peers[(market,t)],utc,ny,fills,usd)

pairs = defaultdict(list)

for (wallet,market,t),activity in own.items():
    baseline = peers.get((market,t))
    if baseline and sum(baseline["utc"][0]) > 0:
        pairs[(wallet,t)].append((activity,baseline))

def summarize(items):
    out = {
        "markets":len(items),
        "fills":int(sum(sum(a["utc"][0]) for a,b in items)),
        "data":{}
    }
    for zone in ("utc","ny"):
        out["data"][zone] = {}
        for i,metric in enumerate(("fills","usd")):
            actual = [0.0]*24
            expected = [0.0]*24

            for a,b in items:
                n = sum(a[zone][i])
                denom = sum(b[zone][i])
                if denom <= 0:
                    continue
                for h in range(24):
                    actual[h] += a[zone][i][h]
                    expected[h] += n*b[zone][i][h]/denom

            na = sum(actual)
            nb = sum(expected)
            out["data"][zone][metric] = {
                "wallet":[round(100*x/na,3) if na else 0
                          for x in actual],
                "comparison":[round(100*x/nb,3) if nb else 0
                              for x in expected]
            }
    return out

profiles = []
repeat_n = 0
repeat_positive = 0

for (address,) in con.execute(
    "SELECT address FROM flagged ORDER BY address"
).fetchall():
    p = {
        t:summarize(pairs[(address,t)])
        for t in ("discovery","validation","all")
    }
    hour = None
    lift = None

    early = p["discovery"]
    late = p["validation"]

    if early["fills"] >= 100 and early["markets"] >= 3:
        a = early["data"]["utc"]["fills"]
        hour = max(
            range(24),
            key=lambda h:a["wallet"][h]-a["comparison"][h]
        )

        if late["fills"] >= 100 and late["markets"] >= 3:
            b = late["data"]["utc"]["fills"]
            lift = round(
                b["wallet"][hour]-b["comparison"][hour],3
            )
            repeat_n += 1
            repeat_positive += int(lift > 0)

    profiles.append({
        "wallet":address,
        "periods":p,
        "discovery_hour_utc":hour,
        "validation_excess_pp":lift
    })

profiles.sort(key=lambda x:-x["periods"]["all"]["fills"])

result = {
    "v4_flagged":int(cal["flagged"]),
    "v4_eligible":int(cal["eligible_wallets"]),
    "mapped_flagged":nflags,
    "cutoff":start,
    "holdout":"2026-09-01",
    "evaluated_wallets":repeat_n,
    "positive_retest":repeat_positive,
    "wallets":profiles
}

backup = root/(
    "web_before_wallet_upgrade_" +
    datetime.now().strftime("%Y%m%d_%H%M%S")
)
shutil.copytree(web,backup)

(web/"data/wallet_timing_v4.json").write_text(
    json.dumps(result,separators=(",",":"))
)

panel = """
<!-- WALLET-TIMING-V4 -->
<h2 class="sectiontitle">
Individual wallet timing · historical holdout
</h2>
<div class="panel">
<div class="hd">
<h2>Does a particular wallet trade at a specific hour?</h2>
<span class="sub">Same-market comparison</span>
</div>
<div class="bd">
<p>Select a historically flagged wallet. Teal bars show
its trading activity. Blue bars show an activity-weighted
comparison with other wallets in the same markets.</p>

<div style="display:flex;gap:12px;flex-wrap:wrap;margin:14px 0">
<label>Wallet
<select id="wt-wallet"></select></label>
<label>Period
<select id="wt-period">
<option value="discovery">June–August 2026</option>
<option value="validation" selected>September–November 2026</option>
<option value="all">All post-cutoff activity</option>
</select></label>
<label>Time zone
<select id="wt-zone">
<option value="utc">UTC</option>
<option value="ny">New York (DST-adjusted)</option>
</select></label>
<label>Measure
<select id="wt-metric">
<option value="fills">Fill frequency</option>
<option value="usd">Fill-side dollars</option>
</select></label>
</div>

<div class="kpis">
<div class="metric">
<div class="val" id="wt-fills">—</div>
<div class="lbl">Wallet fill records</div>
</div>
<div class="metric">
<div class="val" id="wt-markets">—</div>
<div class="lbl">Comparable markets</div>
</div>
<div class="metric">
<div class="val" id="wt-repeat">—</div>
<div class="lbl">Later excess at discovery hour (pp)</div>
</div>
</div>

<p class="small">Teal = selected wallet · Blue = same-market comparison</p>
<div style="overflow-x:auto">
<div id="wt-bars" style="display:flex;align-items:flex-end;
gap:5px;min-width:730px;height:245px"></div>
</div>
<p class="small" id="wt-description"></p>
</div>
</div>

<div class="callout warning">
<p id="wt-overall">Loading wallet-level evaluation.</p>
<p class="small">The most distinctive UTC hour is identified
using June–August trading and checked in September–November.
A wallet must have at least 100 eligible fill records and
three comparable markets in each period to enter the
holdout comparison. This is exploratory, not a formal
significance test. Flagged wallets are not proven insiders,
and some information timestamps are approximate.</p>
</div>
"""

timing = web/"timing.html"
html = timing.read_text()

if "<!-- WALLET-TIMING-V4 -->" not in html:
    if "</main>" not in html:
        raise SystemExit("Could not locate </main>")
    html = html.replace("</main>",panel+"\n</main>",1)

script = r"""<script>
(()=>{
const E=id=>document.getElementById(id);
let D;
const fmt=x=>Number(x).toLocaleString('en-US');

function draw(){
 const w=D.wallets.find(x=>x.wallet===E('wt-wallet').value);
 if(!w)return;
 const p=w.periods[E('wt-period').value];
 const t=p.data[E('wt-zone').value][E('wt-metric').value];
 const mx=Math.max(.1,...t.wallet,...t.comparison)*1.1;

 E('wt-bars').innerHTML=t.wallet.map((v,h)=>`
 <div style="flex:1;height:100%;display:flex;
 align-items:center;flex-direction:column;justify-content:flex-end">
 <div style="width:100%;height:210px;display:flex;gap:2px;
 align-items:flex-end;justify-content:center">
 <div title="Wallet ${v.toFixed(2)}%"
 style="width:40%;height:${100*v/mx}%;background:#24b99e"></div>
 <div title="Peers ${t.comparison[h].toFixed(2)}%"
 style="width:40%;height:${100*t.comparison[h]/mx}%;
 background:#7c9fc7"></div>
 </div>
 <span style="font-size:10px">${String(h).padStart(2,'0')}</span>
 </div>`).join('');

 E('wt-fills').textContent=fmt(p.fills);
 E('wt-markets').textContent=fmt(p.markets);
 const lift=w.validation_excess_pp;

 E('wt-repeat').textContent=lift===null
 ?'Not enough data'
 :(lift>=0?'+':'')+lift.toFixed(2);

 const peak=t.wallet.indexOf(Math.max(...t.wallet));

 E('wt-description').textContent=p.fills
 ?`Peak hour: ${String(peak).padStart(2,'0')}:00–${String(peak).padStart(2,'0')}:59
 ${E('wt-zone').value==='utc'?'UTC':'New York local time'}.
 Wallet: ${t.wallet[peak].toFixed(2)}%;
 peers: ${t.comparison[peak].toFixed(2)}%.
 The holdout value uses the discovery-selected UTC hour,
 which may differ from this peak.`
 :'No qualifying observations for this wallet and period.';
}

async function init(){
 try{
  const r=await fetch('data/wallet_timing_v4.json',
                      {cache:'no-store'});
  if(!r.ok)throw Error('File missing');
  D=await r.json();

  E('wt-wallet').innerHTML=D.wallets
   .filter(w=>w.periods.all.fills>0)
   .map(w=>`<option value="${w.wallet}">
   ${w.wallet.slice(0,10)}… ·
   ${fmt(w.periods.all.fills)} fills</option>`)
   .join('');

  E('wt-overall').textContent=
  `Latest V4 screen: ${fmt(D.v4_flagged)} flagged
  of ${fmt(D.v4_eligible)} eligible wallets.
  ${fmt(D.evaluated_wallets)} wallets satisfy the
  two-period sample thresholds; ${fmt(D.positive_retest)}
  retain positive excess activity at their discovered
  UTC hour in the later period. This is descriptive
  evidence, not proof of informed trading.`;

  for(const id of [
   'wt-wallet','wt-period','wt-zone','wt-metric'
  ]){
   E(id).addEventListener('change',draw);
  }
  draw();
 }catch(err){
  E('wt-overall').textContent=
   'Wallet timing failed to load: '+err.message;
  console.error(err);
 }
}
init();
})();
</script>"""

if 'id="wt-bars"' in html and 'wallet_timing_v4.json' not in html:
    html = html.replace("</body>",script+"\n</body>",1)

timing.write_text(html)

card = f"""
<!-- LATEST-V4-RESULTS -->
<h2 class="sectiontitle">Latest V4 findings · expanded archive</h2>
<div class="kpis">
<div class="metric">
<div class="tag">V4 wallet screen</div>
<div class="val">{int(cal["flagged"]):,}</div>
<div class="lbl">Flagged of {int(cal["eligible_wallets"]):,} eligible wallets</div>
</div>
<div class="metric">
<div class="tag">Evaluated later</div>
<div class="val">{int(ev["n_informed_with_eval"]):,}</div>
<div class="lbl">Flagged wallets with subsequent data</div>
</div>
<div class="metric">
<div class="tag">Expanded training</div>
<div class="val">2,682</div>
<div class="lbl">Prepared markets</div>
</div>
<div class="metric">
<div class="tag">Monthly backtest</div>
<div class="val">22</div>
<div class="lbl">Walk-forward months</div>
</div>
</div>
<p class="small">
The earlier 19-wallet result belongs to a different historical
screen, not this V4 analysis. Wallet flags are not proof of
nonpublic information.
<a href="timing.html">Explore individual wallet timing →</a>
</p>
<!-- END LATEST-V4-RESULTS -->
"""

for filename in ("index.html","findings.html","empirical.html"):
    page = web/filename
    text = page.read_text()

    if "<!-- LATEST-V4-RESULTS -->" in text:
        continue

    anchor = '<div id="hist-kpis" class="kpis"></div>'

    if filename == "index.html" and anchor in text:
        text = text.replace(
            anchor,
            card+'\n<h2 class="sectiontitle">Earlier historical benchmark</h2>\n'+anchor,
            1
        )
    else:
        text = text.replace("<main>","<main>\n"+card,1)

    page.write_text(text)

print("\nSUCCESS: V4 timing and latest wallet counts added")
print("V4 flagged wallets:",cal["flagged"])
print("Mapped wallets:",nflags)
print("Holdout-qualified wallets:",repeat_n)
print("Positive timing retests:",repeat_positive)
print("Website backup:",backup)
print("Open http://localhost:8000/timing.html")
