from pathlib import Path
from datetime import datetime
import json, shutil, zipfile
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

root = Path.cwd()
web = root/'web'
paths = [
 web/'data/walk_forward_evaluation.json',
 web/'data/trader_timing.json',
 web/'data/wallet_timing_v4.json',
 root/'results_v4/validation_report.json',
 root/'data/run_expanded/wallet_backtest.json',
 root/'data/prepared_history_expanded.csv',
 web/'index.html', web/'timing.html',
 web/'empirical.html', web/'tracker.html'
]

for p in paths:
 if not p.is_file():
  raise SystemExit(f'STOP: Missing {p}; export not created.')

for name in ['model.json','wallet_model.json','wallet_state.json']:
 a=root/'data/run'/name
 b=root/'data/run_expanded'/name
 if not a.exists() or not b.exists() or a.read_bytes()!=b.read_bytes():
  raise SystemExit('STOP: Expanded model mismatch: '+name)

wf=json.loads(paths[0].read_text())
wt=json.loads(paths[2].read_text())
v4=json.loads(paths[3].read_text())
wm=json.loads(paths[4].read_text())

if (wf['evaluated_markets']!=2645 or
    wf['n_monthly_folds']!=22 or
    not wt.get('wallets')):
 raise SystemExit('STOP: Backtest or timing data not verified.')

m=wf['metrics']
k=m['with_skill']
b=m['raw_market_price']

flags=v4['events']['calibration']['flagged']
eligible=v4['events']['calibration']['eligible_wallets']
qual=wt['evaluated_wallets']
passed=wt['positive_retest']

if flags!=207 or eligible!=17254:
 raise SystemExit('STOP: Unexpected V4 wallet counts.')

backup=root/(
 'web_backup_before_export_'+
 datetime.now().strftime('%Y%m%d_%H%M%S')
)
shutil.copytree(web,backup)

# Preserve original styling; correct outdated demonstration warning.
p=web/'index.html'
s=p.read_text()
s=s.replace(
 'Live tracker loads demonstration data initially',
 'Real-trained model · public API tracker'
)
s=s.replace(
 '<span class="pill demo">Real-trained model · public API tracker</span>',
 '<span class="pill" style="background:#e6f4ed;color:#086a52;border:1px solid #b7ded0">Real-trained model · public API tracker</span>'
)
s=s.replace(
 'Run the local collector with a real historical model to switch to public API snapshots.',
 'The local collector saves public API snapshots; check timestamps because a snapshot may be cached.'
)
s=s.replace(
 'The initial tracker markets are synthetic test cases, not real events or validated predictions. The read-only public API collector is included in the code package, but genuine live forecasting requires training on the original resolved-trade archive and running the collector on a machine with network access.',
 'When the live collector runs, it uses a model trained on real archived fills. Saved public API snapshots may be stale after disconnection. Historical information timestamps and settlement require independent verification.'
)
p.write_text(s)

# Add expanded results while preserving earlier benchmark sections.
p=web/'empirical.html'
s=p.read_text()

if '<!-- FINAL-EXPANDED -->' not in s:
 section=f'''<!-- FINAL-EXPANDED -->
<h2 class="sectiontitle">Current expanded-model evaluation</h2>
<div class="panel">
<div class="hd">
<h2>22-month walk-forward results</h2>
<span class="sub">2,645 markets · 1,117 events</span>
</div>
<div class="bd">
<div class="tw">
<table>
<thead><tr><th>Model</th><th>AUC ↑</th><th>Brier ↓</th></tr></thead>
<tbody>
<tr><td>PolyInsiders + wallet skill</td><td>{k['auc']:.6f}</td><td>{k['brier']:.6f}</td></tr>
<tr><td>Raw Polymarket prices</td><td>{b['auc']:.6f}</td><td>{b['brier']:.6f}</td></tr>
</tbody>
</table>
</div>
<p class="small">
Wallet-model AUC {wm['test_auc']:.4f} in a separate holdout.
V4 flagged {flags} of {eligible:,} eligible wallets.
Trader timing: {qual} wallets met two-period criteria;
{passed} showed positive later excess at the earlier selected hour.
The wallet-skill effect is inconclusive; flagged status
does not imply private information.
</p>
</div></div>'''
 s=s.replace('<main>','<main>'+section,1)
 s=s.replace(
  '<h2 class="sectiontitle">Data preparation</h2>',
  '<h2 class="sectiontitle">Earlier 1,175-market benchmark</h2><h2 class="sectiontitle">Data preparation</h2>',
  1
 )
 p.write_text(s)

docs=root/'docs'
docs.mkdir(exist_ok=True)

lines=[
 'PolyInsiders | ECE 473 — Updated Final Research Summary',
 'Research: wallet performance persistence, market forecasting, and trader timing.',
 'Archive: 35,583,634 fill-side records; 3,713 markets.',
 'Prepared: 2,682 markets; 383,170 wallet-market observations; 179,824 unique wallets.',
 f"Walk-forward: {wf['evaluated_markets']:,} markets, {wf['evaluated_events']:,} events, {wf['n_monthly_folds']} months.",
 f"PolyInsiders with wallet skill: AUC {k['auc']:.6f}, Brier {k['brier']:.6f}, log loss {k['log_loss']:.6f}.",
 f"Polymarket baseline: AUC {b['auc']:.6f}, Brier {b['brier']:.6f}, log loss {b['log_loss']:.6f}.",
 f"Wallet-skill Brier difference: {wf['brier_with_minus_without']:.8f}. 95% CI {wf['bootstrap']['brier_with_minus_without_ci95']}.",
 f"Separate wallet-model test AUC: {wm['test_auc']:.4f}; {wm['test_wallets']} wallets.",
 f"Aiden V4 screen: {flags}/{eligible:,} flagged, {v4['events']['evaluation']['n_informed_with_eval']} with later evaluation data.",
 f'Exploratory timing: {qual} wallets meet June–August vs September–November criteria; {passed} show positive later excess at discovery-selected UTC hour.',
 'Snapshots are approximately 24 hours before recorded information time with 48 hours of preceding fills. Many information times are scheduled-end proxies.',
 'No reliable wallet-skill incremental effect is established. Neither V4 flags nor timing patterns prove insider trading.',
 'The website uses local public API collection; localhost alone is not a publicly hosted website.',
 'Reproduction requires the companion model-ready CSV; the raw archive is held separately by the team.'
]

(docs/'FINAL_RESEARCH_SUMMARY.txt').write_text(
 '\n\n'.join(lines)
)

styles=getSampleStyleSheet()
pdf=docs/'PolyInsiders_Updated_Report.pdf'

SimpleDocTemplate(
 str(pdf),pagesize=letter
).build(
 [Paragraph('<b>'+lines[0]+'</b>',styles['Title']),
  Spacer(1,12)]+[
  Paragraph(x,styles['BodyText']) for x in lines[1:]
 ]
)

# Create an editable presentation in original cream/green colors.
prs=Presentation()
prs.slide_width=Inches(13.333)
prs.slide_height=Inches(7.5)

def slide(title,bullets):
 s=prs.slides.add_slide(prs.slide_layouts[6])
 s.background.fill.solid()
 s.background.fill.fore_color.rgb=RGBColor(247,248,244)

 def txt(value,y,size,green=False):
  box=s.shapes.add_textbox(
   Inches(.85),Inches(y),Inches(11.8),Inches(1.3)
  )
  p=box.text_frame.paragraphs[0]
  p.text=value
  p.font.size=Pt(size)
  p.font.color.rgb=(
   RGBColor(11,118,93) if green else RGBColor(19,48,45)
  )
  p.font.bold=size>25

 txt('POLYINSIDERS · ECE 473',.35,13,True)
 txt(title,1.05,31)
 for i,line in enumerate(bullets):
  txt('• '+line,2.55+i*.87,18)

slide('Can past wallet performance predict the future?',[
 'Public-market information, not private-data access',
 'Market outcome model and future-wallet-performance model',
 'New: individual-wallet trading-time persistence'
])
slide('Expanded dataset',[
 '35.6 million fill-side records, 3,713 markets',
 '2,682 model-ready markets, 179,824 wallets',
 'Time-restricted snapshots and event-disjoint testing'
])
slide('22-month walk-forward results',[
 f"PolyInsiders AUC {k['auc']:.6f} · Brier {k['brier']:.6f}",
 f"Market baseline AUC {b['auc']:.6f} · Brier {b['brier']:.6f}",
 'Wallet-skill increment is too small to establish a benefit'
])
slide('Wallet analysis and V4 screen',[
 f"Predictive wallet model AUC: {wm['test_auc']:.4f} on {wm['test_wallets']} wallets",
 f'V4 flags: {flags} of {eligible:,} eligible wallets',
 'Statistical flags are not proof of insider trading'
])
slide('Trader timing: new contribution',[
 'UTC and New York hour-of-day / before-event patterns',
 'Same-market comparisons with ordinary trading activity',
 'Discovery June–August; validation September–November'
])
slide('Timing persistence test',[
 f'{qual} flagged wallets met minimum sample criteria',
 f'{passed} retained positive excess at their selected UTC hour',
 'This is exploratory, not a significance test'
])
slide('Live tracker and limitations',[
 'Public-API forecasts saved with timestamps',
 'Scheduled event end is not verified first public news',
 'Future work: audit announcements and score resolved forecasts'
])

ppt=docs/'PolyInsiders_Updated_Presentation.pptx'
prs.save(ppt)

(root/'README_FINAL.md').write_text(
 '# PolyInsiders ECE 473 — Final expanded project\n\n'
 'Preview: python -m http.server 8000 --directory web\n\n'
 'Live collector: python -m polyinsider.cli serve --live --interval 120 --port 8000\n\n'
 'Pages: index.html, tracker.html, empirical.html, timing.html.\n\n'
 'The core ZIP contains code, models, web data, results, '
 'slides and research documents. The companion ZIP contains '
 'the expanded training CSV. Raw Parquet is not included. '
 'V4 screening and the predictive wallet model are separate analyses.\n'
)

# Package only relevant project files, not massive raw archives.
out=root.parent/'PolyInsiders_ECE473_FINAL_UPDATED.zip'
files=[]

for folder in [
 'web','polyinsider','docs','slides',
 'data/run','data/run_expanded',
 'data/walk_forward_evaluation',
 'data/walk_forward_evaluation_expanded'
]:
 d=root/folder
 if d.is_dir():
  files.extend(
   p for p in d.rglob('*')
   if p.is_file()
   and p.suffix.lower() not in ['.parquet','.zip']
   and p.stat().st_size<80_000_000
   and '__pycache__' not in p.parts
  )

files += [
 p for p in root.iterdir()
 if p.is_file()
 and p.suffix.lower() in ['.py','.md','.txt','.sh']
]

files += [
 root/'results_v4/validation_report.json',
 root/'results_v4/likely_informed_wallets.csv',
 root/'data/prepared_history_expanded.import_report.json'
]

with zipfile.ZipFile(
 out,'w',zipfile.ZIP_DEFLATED,compresslevel=6
) as z:
 for p in sorted(set(files)):
  if p.is_file():
   z.write(p,p.relative_to(root).as_posix())

with zipfile.ZipFile(out) as z:
 assert z.testzip() is None
 for p in [
  'web/timing.html',
  'web/data/wallet_timing_v4.json',
  'data/run/model.json',
  'docs/PolyInsiders_Updated_Presentation.pptx',
  'docs/PolyInsiders_Updated_Report.pdf'
 ]:
  assert p in z.namelist(),p

# A companion ZIP lets us provide training data separately.
comp=root.parent/'PolyInsiders_ECE473_MODEL_READY_DATA.zip'

with zipfile.ZipFile(
 comp,'w',zipfile.ZIP_DEFLATED,compresslevel=6
) as z:
 z.write(
  root/'data/prepared_history_expanded.csv',
  'data/prepared_history_expanded.csv'
 )

with zipfile.ZipFile(comp) as z:
 assert z.testzip() is None

print('\nSUCCESS: Final website, report, slides and ZIPs verified.')
for p in [out,comp,pdf,ppt,backup]:
 print(p)
print(f'Timing holdout: {passed}/{qual} positive retests')
