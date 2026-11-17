# Project handoff: likely-informed traders on Polymarket (ECE 473)

Updated November 13, 2026 (with a newer summary and sections 15 to 20 added November 16), for Emilly Antero and her assistant (ChatGPT). Read this whole file first. It explains where the project stands, what the code does, what the results are and are not, and what is left. The zip that comes with it **includes the data** (about 1 GB) but not the private key; see section 8.


---
## READ FIRST: where things stand on November 16, 2026 (written for Emilly)

Sections 1 to 14 below were written on Nov 13 and earlier. Sections 15 to 21 at the end are newer and **override the earlier numbers where they differ**. In plain terms:

**What was done on Nov 16 (by Aiden with Claude):**
1. **More data.** Downloaded Jan to Dec 2025 (1,508 markets) and Oct to Nov 2026 (595 markets) and combined them with the 1,610 markets from Jan to Sep 2026: **3,713 markets, 35.6 million trades, 1.06 million wallets.** Total profit across all wallets is -$465 on $5.7 billion traded, so both sides of every trade are captured. Cleaned data is in `data_events_full/`.
2. **Re-ran the wallet flagging on the full history** (`results_v3/`). The original rule flagged only 23 of 17,254 eligible wallets because it assumed more randomness than the data has. A less conservative version of the same test (`results_v4/`) flags 207 wallets, and those wallets, including the roughly 180 the old rule missed, still beat matched wallets afterwards. Simulation: no false flags with pure noise; recall of planted informed traders rises from about 40% to about 83% at 96% precision. See section 18 for caveats (the placebo still shows some persistence; 30 flagged wallets traded under $1,000).
3. **Tested whether we can predict informed trades as they happen (the alerts tab).** Added three families of features (bot filter, "other good wallets agree", trader behaviour) and tested the alert model on five non-overlapping quarters. **Result: the machine-learning alert model does not work out of sample** (the old model is -21% pooled; the agreement version is +13% but the interval, -15% to +47%, spans zero and the result swings from -99.8% to +54% between quarters). A simple rule with no model, **"copy trades by wallets whose past score is at least 3 over 10 or more events"**, is the only strategy whose pooled interval is above zero: **+8.1% per dollar (CI +1.6% to +14.6%)** vs -3.7% for random trades (section 19).
4. **One-shot test on the fresh Oct to Nov 2026 window** (never used for any decision): the frozen rule earned **+8.4% (CI -0.8% to +18.0%)** at a 5-minute delay, but random trades also earned +4.1% in that window, so the edge over random is only about 4 points and the interval includes zero. Same size as the earlier windows, not significant on its own (section 20).

5. **Trade timing** (section 21): flagged wallets trade earlier than other wallets (more in the first 10% of a market's life and 90+ days before resolution), not at the last minute, and show bursts in the 10 minutes before sharp price jumps (3 times the normal share, 70% of those dollars on the side the price then moved to), with no edge after the jump. The signal comes from a subset of large trades and cannot separate information from price impact.

**What this means for the website and report.** The Alerts tab still shows the old model-based results and must be reworked around the simple rule; the negative result for the model should be stated plainly. The site and notebook still read the 2026-only results (`results_v2`). Nothing on them has been switched to the full-history numbers yet.

**Honest bottom line.** There is evidence that some wallets keep beating prices, and that following wallets with a strong past record has earned a small positive return (about 8% per dollar before real-world costs such as liquidity and fees, 4 of 6 quarters positive). The evidence is modest, not proof, and does not show private information. The agreement idea and the machine-learning alert model should not be presented as results.

**The walkthrough notebook (`polyinsider_walkthrough.ipynb`) was updated:** steps 1 to 16 are unchanged (2026-only data) except for notes that point to the newer results; a new **Part B (steps 17 to 21)** covers the full history, the re-flagging, the five-quarter forward test, the fresh-window test and the updated bottom line. Part B reads saved result files and runs in seconds; the whole notebook runs top to bottom in about 3 minutes and is saved with its outputs.

**New code:** `run_alert_windows.py` (forward-window test), `run_alert_fresh.py` (one-shot fresh-window test), `combine_market_lists.py`, `cloud_run.sh` and `adroit_analyze.slurm` (untested options for big runs), `pm473/alerts.py::add_context_features`, `run_flagging_v2.py --sigma-floor 0 --center`. Tests: 54 pass (`python -m pytest -q tests`). The raw trade cache `data/` (1.8 GB) and the large alert-candidate files are **not** in the zip; rebuild candidates with `python build_alert_candidates.py --data data_events_full/live --results results_v3` (about a minute).
---

## 1. The project in plain terms

Polymarket is a prediction market. People trade contracts on real events (elections, Fed decisions, inflation reports). A contract pays $1 if the event happens and $0 if not, so its price is the market's probability estimate.

**Question:** can we find wallets (trader accounts) whose results beat the market's prices by more than luck explains, and do those wallets keep doing better afterwards? We call them **likely informed**. We do **not** claim they have private information or broke any rules; our evidence can only show persistent outperformance.

**Current plan:**
1. Define precisely how a wallet is flagged, and validate that definition.
2. Calculate flagged wallets' profit and loss (PnL) and return per trade.
3. Show their trades on a website ("Polyinsider"), ideally live, with the flags trained on history.

## 2. Instructor feedback (Nov 10) and what it changed

From Niusha Moshrefi:
- The Milestone I report read as AI-generated and was hard to follow. Example: it said "PnL" without defining it and described a bug fix that did not belong in a one-page progress report. **Future writing: plain language, define every term, report only what matters.**
- The scope was too big. Identifying informed traders and properly testing the method is enough. Trade sizing, price-impact models and copy-trading are out of scope.
- Clearly define and validate the method that labels wallets informed or uninformed.
- Polymarket may not provide historical order books; how is that handled? (It does not. Order books were cut, so no longer needed.)

Cut from scope: copy-trading agent, prediction algorithm, historical order books. The game-theory model is cut or reduced to a short discussion section; **Emilly's role is undecided (section 10).**

## 3. Glossary

- **Wallet:** one trader's account address. Trades are public on the blockchain.
- **Trade (fill):** one executed purchase or sale.
- **Market:** one yes/no question. **Event:** a group of related markets (one election, one market per candidate).
- **Resolve:** the market settles; a winning share pays $1.
- **PnL (profit and loss):** dollars made or lost, after fees.
- **Return per trade:** (payout minus price paid) / price paid. A 10-cent outcome that wins returns +900%; if it loses, -100%.
- **Evidence score:** how far a wallet's profit sits above what luck at fair prices would give. About 2 or more is notable; it is only comparable between wallets with similar numbers of events.
- **Cut-off (split) date:** June 1, 2026. Wallets are scored on events that resolved before it and checked on events after it. Only the second step is evidence the skill is real.
- **Placebo:** crypto "up or down" markets, decided by a public price feed, so private information is not possible. We run the same test there for comparison.
- **Rank correlation:** how well ordering wallets by earlier score matches ordering by later results. 0 means no relationship, 1 means perfect.
- **AUC:** the chance that a randomly chosen wallet that beat prices later had a higher earlier score than one that did not. 0.5 is a coin flip.
- **Benjamini-Hochberg (BH):** the standard correction for testing thousands of wallets at once, which otherwise lets lucky wallets through. We set it to allow 10% false flags.
- **Confidence interval (CI):** range of values consistent with the data. If a 95% CI includes zero, the result is not statistically distinguishable from no effect.

## 4. Data

- Source: Polymarket's public Data API v2 and Gamma API. No key needed.
- **Events sample:** 1,610 markets that resolved Jan 1 to Sep 30, 2026 (elections, economics, Fed), each over $50,000 volume. 15.6 million trades, 548,066 wallets, about $2.55 billion volume. 883 markets before the cut-off, 727 after.
- **Placebo sample:** 300 crypto up/down markets, 2.9 million trades, 105,209 wallets, about $32 million volume (160 before the cut-off, 140 after).
- **Checks done:** 200 of 200 random trades matched the blockchain record. Total profit across all wallets is about $0 (-$465 on $2.55B), as it should be since every trade has two sides.
- **Limits:** 92% of markets use the scheduled end date as the time information became public (only 72 Fed markets have exact times); trades that convert positions between outcomes of one event are not in the data; no historical order books.

## 5. The flagging method (current)

An earlier version took each wallet's average return per event, ran a t-test and corrected with BH. It was too lenient: it flagged 4.5% of wallets on the placebo, where almost none should be flagged (long-shot returns are extremely skewed and wallets in the same markets are not independent). The current version, in `pm473/skill_test.py`:

1. If prices are fair, a bet at price p wins with probability p, so on average it earns zero. Score each wallet's profit in each **event** against that benchmark, scaled by how much luck alone would vary it.
2. Cap any single event's score at +/-3 so one lucky long shot cannot decide a wallet.
3. Combine a wallet's event scores into one evidence score. Require at least **10 events** (earlier version: 5).
4. Compare the spread of scores across all wallets with theory; if wider, shrink every score by that factor.
5. Convert to p-values and apply BH at 10%. A stricter rule (Benjamini-Yekutieli) is run as a sensitivity check.

## 6. Results (run November 11; flag before June 1, test after)

Do not use the old numbers (1,200 flagged; +7.8 points); they came from the lenient test.

- **False flags fixed:** the placebo flags 0.53% of wallets (21 of 3,944), down from 4.5%. A simulated world with no skilled traders flags 0.
- **Few wallets flagged:** 19 of 4,009 eligible wallets on events (0.47%). Using all history (what the live page uses), 29 of 9,634.
- **The full score predicts later results, much more on events than on the placebo.** Rank correlation between earlier score and later result: **0.24** (95% CI 0.20 to 0.28) on events vs **0.07** (0.02 to 0.12) on the placebo. AUC 0.65 vs 0.55 (volume alone: 0.50 on events). On events, 0.21 to 0.25 in each of three time windows; placebo 0.07 to 0.13.
- **Of the 19 flagged wallets scored later, 95% beat prices vs 57% of other wallets** (Fisher p = 0.0004). The placebo's 21 flagged wallets: 100%.
- **Flagged vs matched wallets (later event score):** +0.40 (CI +0.19 to +0.61) on events, but +0.46 (CI -0.10 to +1.07) on the placebo. Only about 20 wallets per group, so the intervals are wide and the two samples cannot be told apart on this measure. Over 36 settings (cut-off date, minimum events, correction, strictness): events positive in 36 (interval above zero in 35); placebo positive in 34 (above zero in 5).
- **Important caveat:** the top-scoring group does equally well later in both samples (+0.13 vs +0.13). The events ranking is stronger mainly because low scorers keep underperforming there (-0.08 vs +0.02). So the evidence is persistent outperformance, which could come from information, speed or other skill. It does **not** show private information, and the placebo cannot separate these.
- **Wallet types:** some flagged wallets look like bots or market makers (many events, tiny profit per dollar traded). Report PnL per dollar alongside the score.
- **Simulation check (offline, any laptop):** with 60 planted skilled traders the method finds roughly a third to 40% of them and every flag is correct; the stricter rule finds almost none.

Full numbers with plain-language meanings: `results_v2/report/key_numbers.md`.

## 7. Dashboard and live feed

- **`site/` + `build_dashboard.py`:** a multi-page dark-theme site (Home, Wallets, Live feed, Findings, Method, About). Flagged wallets show an amber row and FLAGGED tag next to the top near-misses; evidence-score bars mark the flag cut-off; clicking a wallet opens its trade history with return and PnL. The Findings page shows all seven writeup figures with captions, downloads and the tables; the Method page pulls its numbers from the saved results. Rebuild with `python build_dashboard.py --data data_events/live --results results_v2`; view with `python -m http.server -d site 8000`.
- **`live_feed.py`:** watches the 29 all-history flagged wallets and writes new trades to `site/data/live_feed.json`; the page refreshes it every 15 seconds. It filters by wallet using the `user` parameter of the Data API (confirmed working by a probe on Nov 11). Open trades show no return until their market resolves; returns ignore fees.
- **Open risks:** markets created from about Nov 2 use newer contracts that the main pipeline excludes, so live coverage of brand-new markets should be spot-checked; labels go stale and must be re-run as markets resolve; the live tab is a demonstration, not part of the validated result.
- Wording on the page is "likely informed" with a disclaimer. The name "Polyinsider" implies wrongdoing, so the report should explain the choice.

## 8. What is in the zip, and what is not

**This zip (FULL, Nov 12) includes the cleaned data** (`data_events/live`, `data_placebo/live`) and `results_v2/`, so everything below can be re-run, including the notebook and the website build. It is for a fresh machine (Emilly's). **Do not unzip it over Aiden's own `~/pm473`.** Older, code-only zips were meant for that.

Included: all code, tests, the page, `README.md`, `.env.example`, and `results_v2/report/` (7 figures as PNG and PDF, 5 tables, captions, key numbers), `results_v2/eval/`, `results_v2/validation_report.json`, the two small lists of flagged wallets, and `archive_before_nov11/` (older work).

**Not included, on purpose:** `.env` (holds a private key; the notebook and website do not need it) and `data/live` (an older, larger download not used by the current workflow). The live feed needs internet only, no key.

```
pm473/                    code package (skill_test.py, model_eval.py, trade_returns.py are the newest)
run_milestone1.py         downloads and cleans data (needs data folders)
run_flagging_v2.py        flags wallets, runs all validation checks     -> results_v2/
evaluate_model.py         out-of-sample performance statistics          -> results_v2/eval/
make_report_figures.py    figures, tables, captions, key numbers        -> results_v2/report/
build_dashboard.py, live_feed.py, site/    the website and live feed
tests/                    31 automated tests
archive_before_nov11/     older files (see its README)
```

## 9. Running things

Set up (any computer, Python 3.9 or newer):
```bash
cd pm473
python3 -m venv ~/pm473_env && source ~/pm473_env/bin/activate
pip install -r requirements.txt
python -m pytest -q tests                      # expect 31 passed
```
Works without the data:
```bash
python -c "import run_flagging_v2 as r, json; print(json.dumps(r.synthetic_validation(1), indent=1))"
```
With the data (Aiden's laptop), in this order:
```bash
python run_flagging_v2.py --data data_events/live --placebo data_placebo/live
python evaluate_model.py
python make_report_figures.py
python run_flagging_v2.py --data data_events/live --label-all
python build_dashboard.py --data data_events/live --results results_v2
jupyter notebook polyinsider_walkthrough.ipynb   # the step-by-step walkthrough (see below)
python live_feed.py                            # own terminal
python -m http.server -d site 8000             # own terminal, then open localhost:8000
```
Note: on Aiden's Mac the conda Python crashes on import, so he uses the venv above.

**Data safety (Aiden's laptop):** never delete, move or replace anything in `data*`; update code by unzipping a code-only zip from the folder that holds `pm473` with `unzip -o`, never by replacing the folder in Finder; back up the data folders after every download or analysis. A full day of downloads was lost on Oct 28. Never put API keys anywhere except `.env`.

## 10. What is left and open decisions

- **Write up the results** in plain language, using `results_v2/report/` (figures, captions, key numbers). State plainly that the evidence is persistence, not private information.
- **Emilly's role in Milestone II is undecided.** The game-theory model is cut; a natural replacement is leading the validation write-up and the sensitivity discussion. Needs agreement.
- Confirm what exactly is due on 11/12 (the Milestone I instructions asked for the remaining work to be done by then).
- Decide whether the live feed is required or a stretch demo (recommendation: stretch; the historical page is the committed deliverable).
- Optional: exact release times (8:30 am ET) for inflation and jobs-report markets to sharpen information timing.

## 11. Guidance for ChatGPT

- Plain language, short sentences, define every term the first time, no filler.
- Do not describe internal bug fixes in the report; say what the method does and how it was validated.
- Say "likely informed" or "persistently beat market prices", never "insiders", and never claim private information.
- Always give the interval and the number of wallets behind a number. Never invent numbers; if a result is not here or in `results_v2/report/`, ask Aiden to run it.
- Keep Milestone submissions to one page when asked.
- Reference papers from the original one-pager: Chen et al. (2010) on gaming prediction markets with a market maker; Dubach (2026) on Polymarket's order book vs on-chain record; Kyle (1985) on informed trading.

## 12. Update, Nov 12: notebook, sensitivity and index results

- **`polyinsider_walkthrough.ipynb`** runs every step with an explanation (flagging, false-flag checks, out-of-sample results, accuracy and PnL for informed vs random vs average wallets, sensitivity, one wallet's returns). Run top to bottom in the `pm473_env` virtual environment (VS Code: select that interpreter as the kernel; or `jupyter notebook`). It uses the real data if present, otherwise simulated data. Outputs saved in the file are from Aiden's data. In the notebook the main (scheduled-events) sample is labeled **"main sample"** (setting `MAIN_LABEL`); saved files and code still call it `events`.
- **Strictness `q` (step 8b).** q = 10% is the headline. At q = 5% the test finds very few skilled traders even in simulation. At q = 20% to 30% the main sample's advantage shrinks (+0.40 to about +0.25) and the placebo's flags grow faster (21 to 66 at q = 30%, against 19 to 33), and the placebo's flagged wallets also beat matched wallets later. Loosening q does not separate the main sample from the placebo any better. In simulation, recall at q = 10% is about 24% to 42% depending on seeds, with precision near 100%.
- **Informed-activity index (step 12, exploratory, not on the site).** Share of daily dollars traded by flagged wallets, and the direction of their net trading, tested out of sample against 200 random matched groups. Result: flagged wallets trade about 1.75% of daily dollars; the next-day edge is slightly negative (-0.4 cents per share, interval -0.8 to -0.1); the outcome edge is about 0. **No predictive value shown, so do not add the index to the website.** A possible reason for the gap with the PnL result: flagged wallets may profit through better prices (market-maker style), not direction.
- **To update the website:** unzip, set up the environment (section 9), then `python build_dashboard.py --data data_events/live --results results_v2` and `python -m http.server -d site 8000`. The site's wording still says "events" in places; change it to "main sample" if the report does.

## 13. Update, Nov 13: alert model, lead-time stats, copy-trading test, Alerts tab

The TA asked for a model that flags informed trades **before the market resolves**, stats on **how long before resolution** wallets trade, and a test of **how much you could make trading with them**. Done on historical data only (train earlier, test later).

- **Code:** `pm473/alerts.py` (causal features, model), `pm473/alert_backtest.py` (copy-trading test), `build_alert_candidates.py` (step 1, about 15 s), `evaluate_alerts.py` (step 2, about 25 s per cut-off; also `--cutoff`, `--merge`, `--live-model`), `make_alert_figures.py` (figures), `live_alerts.py` (live scorer), `site/alerts.html` (the tab), `tests/test_alerts.py` (9 tests; 40 total). Outputs are in `results_v2/alerts/`. Notebook steps 13 to 16 walk through it.
- **Design.** Each trade is scored with only what was known then: the wallet's history from events already resolved, the price, the size, and hours to the market's **scheduled** end. Label: edge = payout minus price paid, per share. Train on trades before June 1 (in events resolved by then), test after. Four models: all features, trade features only, wallet history only, price only (the benchmark for "no wallet information"). Alert threshold is fixed on the last fifth of the training period.
- **A leak was found and removed.** The cleaned data's `hours_to_info` falls back to the realised resolve time for markets that resolved before their scheduled end. The model now uses the scheduled time instead. Do not use `hours_to_info` as a model input.
- **Copy-trading rules.** $100 flat per alert, one alert per wallet, market and side, price paid = next trade in that market after a 0 min / 5 min / 1 hour delay (no historical order books), held to resolution, intervals resample whole events, compared with the same number of random trades.
- **Results (train before June 1, test after; 5 minute delay):**
  - Trained model (all features): +3.9% per dollar (95% interval -11% to +18%); across training cut-offs of May 1, June 1, July 1 it returned -32%, +4%, -2%. **Not reliable; not better than the price-only benchmark** (+5%).
  - Trading with the 19 wallets flagged before the cut-off: +7.8% (-3% to +19%); +6%, +8%, +20% across the three cut-offs. Past-score rule (past Z of at least 3, 10+ events): +8.0% (-1% to +18%); +7%, +8%, +19%. Positive in all three windows and above the random-trade band, but intervals mostly include zero, and the running return was negative until a few large late events resolved. **The profit is concentrated: one wallet supplies 88% of the flagged-wallet profit in the main window; without the most profitable wallet the flagged-wallet return is +1%, +3%, +12% and the past-score rule's is +5%, +5%, +14% (all intervals include zero).** **Suggestive at best.** One extra cent per share of cost roughly halves it.
  - **Lead time:** by trade count the flagged wallets trade at about the same time as everyone else (median about 12 days before resolution). Their dollars arrive earlier (median about 26 days against 5), **but one wallet places about 89% of their dollars, mostly in a single event; without it the figure is about 9 days.** Do not claim flagged wallets trade early in general.
- **Alerts tab** (`site/alerts.html`): bottom line written from the data, backtest table (switch delay and extra cost), figures, lead-time table, replay of alerts with outcomes (past-score rule or model), and live alerts. **Live alerts are untested against the real Data API from the build sandbox** (no network there); the scoring logic is unit-checked with made-up trades. Run `python evaluate_alerts.py --live-model` first, then `python live_alerts.py`. The saved model file may need re-creating if scikit-learn versions differ (same command).
- **Not on the site:** the informed-activity index (step 12 of the notebook; no predictive value).
- **Wording:** "alerts" are trades that looked likely to pay off from past results and price; never "insider trades", never advice. Report intervals and the number of alerts and wallets with every number.

## 14. Plan: more data, then the stronger alert ideas (started Nov 16)

**Why.** Only about 20 wallets are flagged and one supplies most of the copy-trading profit. More history and a fresh test window are the main fix. Candidate improvements, in order: (1) drop market-maker/bot-like wallets, (2) add agreement between good wallets, (3) add behaviour features, (4) price move after a trade as a second label, (5) release-time study for Fed and data markets. **Protocol:** build and tune on data already seen (time-based validation inside the training period only), **freeze the design, then evaluate once** on the new 2025 history and on the fresh October to November 2026 window. Do not keep adjusting after looking at the fresh window.

**New data (do not touch the existing `data*` folders; new folders only).** Downloads need internet, so run them on the laptop (use `caffeinate -i`) or on `adroit-vis` inside `tmux` if the account is active. Compute nodes have no internet. The trade files go into the shared cache `data/live/trades_raw` (resumable, re-run the same command to retry failures). Tags are the same as the original events run.
```bash
# fresh test window first (smaller, most important). Tags are typed inline: a $TAGS variable is NOT split into words by zsh (the Mac default shell).
python run_milestone1.py --out data_events_fresh --trades-dir data/live/trades_raw --end-min 2026-10-01 --end-max 2026-11-15 --n-markets 3000 --stage download --tags fed fed-rates macro-fed federal-reserve elections global-elections world-elections primary-elections primary-election special-elections mayoral-elections economy economic-policy us-economy inflation cpi jobs-report unemployment gdp
# more history: all of 2025
python run_milestone1.py --out data_events_2025 --trades-dir data/live/trades_raw --end-min 2025-01-01 --end-max 2025-12-31 --n-markets 6000 --stage download --tags fed fed-rates macro-fed federal-reserve elections global-elections world-elections primary-elections primary-election special-elections mayoral-elections economy economic-policy us-economy inflation cpi jobs-report unemployment gdp
```
Markets from about Nov 2 use new contracts and are excluded automatically (`version == v1` only). The 2024 US election is left out on purpose (very large, different regime); add later if wanted. Back up the data folders after each finished download (section 1 of the old handoff).

**Analysis over everything (offline).** Combine the saved market lists, then analyse with `--market-list` so no internet is needed:
```bash
python combine_market_lists.py data_events/live data_events_2025/live data_events_fresh/live --out data_events_full_markets.parquet
sbatch adroit_analyze.slurm      # on Adroit, 96 GB, a few hours; or run its two python commands on a laptop with enough memory
```
Memory: the original analysis peaked near 19 GB before fixes; three times the data may need 40 GB or more, which is why Adroit is recommended for `--stage analyze`. The alert code (`build_alert_candidates.py`, `evaluate_alerts.py`) reads in chunks and is light. To use Adroit: copy the code (the code zip, no data), `data/live/trades_raw` (several GB, use Globus or scp) and `data_events_full_markets.parquet`; create the environment once (`conda create -n pm473_env python=3.11`, then `pip install -r requirements.txt`); `sbatch adroit_analyze.slurm`; copy `data_events_full/live` and `results_v3` back. Check Princeton Research Computing's current guidance for module names and limits.

**Cloud VM option (alternative to Adroit).** `cloud_run.sh` runs the downloads and the analysis on one rented Linux machine, then packs the cleaned data and results to pull back. Use `pm473_code_only_20261116.zip` (no data, no `.env`). It is untested on a real VM; read it first. Rough sizing: 16 vCPU, 128 GB memory, 250 GB disk, on-demand for the analysis; delete the VM and disk afterwards. Never copy `.env` to a cloud machine (the on-chain audit is off, so no key is needed).

**Download progress (added Nov 16).** The download stage now shows one live line: bar, percent of trading volume, markets done, rows this run, elapsed time and a rough ETA (by volume; it steadies after a few dozen markets). It also appends one line per finished or failed market to `data/live/trades_raw/_download_log.txt` (watch with `tail -f`). When output is redirected to a file it prints a plain line every 30 seconds instead.

## 15. Alert ideas 1-3 (Nov 16): bot filter, agreement, behaviour (2026 seen data only)

All three downloads finished (3,713 markets, no missing trade files; the six large 2025 markets succeeded on retry). `data_events_full_markets.parquet` holds the combined list. The combined analysis and `results_v3` are run by Aiden on the Mac.

**Code.** `pm473/alerts.py`: `add_context_features(c)` adds 17 causal features to a candidate table (`BOT_FEATURES`, `BEHAVIOUR_FEATURES`, `AGREE_FEATURES`). Each uses only earlier qualifying trades in the table, and no label (tests: `tests/test_alerts_context.py` check that cutting the table at a time gives identical features, and that flipping the labels changes nothing). `evaluate_alerts.py --ctx --cutoff ...` runs the comparison into `results_v2/alerts_ctx/` (nothing existing is overwritten); `--ctx --merge` combines cut-offs. 48 tests pass.

**Results at cut-offs 2026-05-01 / 06-01 / 07-01 (5 min delay, $100 flat, no extra slippage; random trades about -1% to -2.5%).**

| strategy | 05-01 | 06-01 | 07-01 |
|---|---|---|---|
| All features (old model) | -32% | +3.9% | -2.5% |
| Price only | no alerts | +5.3% | -1.3% |
| + bot features | -18% | +3.3% | -11% |
| + behaviour | -26% | +13.5% | -19% |
| + agreement | +25.8% | +16.0% (CI +2% to +28%) | +10.4% |
| + all context | +8.8% | +16.0% | +10.2% |
| + all context, bot-like removed | +8.8% | +16.2% | +10.6% |

Reading: **agreement** (earlier dollars from other good wallets on the same side of the same market) is the only idea that helps in all three cut-offs, with very low profit concentration (top wallet 0.5-9% of profit). Only the 06-01 interval excludes zero; the 05-01 and 07-01 intervals include it, and the 07-01 test has only about 22 events. **Behaviour** features are unstable (sign changes). **Bot features and the bot-like filter** do nothing useful; "two-sided" is confounded because selling flips the held outcome, so it partly measures closing positions. Alerts are mostly favourites (mean price paid about 0.65, win rate about 75%), so a "holds up on new data" claim is NOT yet made.

**Not done / next.** (a) Rebuild candidates on the full history (`build_alert_candidates.py --data data_events_full/live --results results_v3`) and re-run `--ctx` over more cut-offs. (b) Ideas 4 (price move after a trade) and 5 (release-time study). (c) Freeze the design, then evaluate once on the fresh Oct-Nov 2026 window and 2025. The fresh window was not used here.

## 16. Full-history run (Nov 16): 3,713 markets, Jan 2025 to Nov 2026

**Data.** `data_events_full/live/` = cleaned fills for all 3,713 markets (35.6M fills, 1.06M wallets; zero-sum check -$465 on $5.69B). `data_events_2025/live` and `data_events_fresh/live` hold only the market lists of the two new download windows (trade files are in the shared cache `data/live/trades_raw`, which is NOT in the zip). `data_events_full_markets.parquet` = combined list. `results_v3/` = flagging on the full history (split 2026-06-01, min 10 events, BH q=0.10).

**Flagging results (results_v3/validation_report.json).** 17,254 eligible wallets, only 23 flagged (0.13%; the earlier 2026-only run flagged 19 of 4,009). Out-of-sample, the 22 flagged wallets with evaluation events beat matched wallets: mean z +0.28 (CI +0.09 to +0.48), profit per dollar +0.46 (CI +0.18 to +0.79); pooled profit per dollar 9.0% vs 1.8% for the rest. **Caveats:** (1) the placebo (crypto up/down) flags 0.53% of wallets, more than the events sample, and its flagged wallets have a similar out-of-sample mean z (+0.46, CI -0.10 to +1.07, 8 wallets); only profit per dollar separates (events +0.46, placebo +0.03). (2) The flagged group is tiny and concentrated: one wallet holds about 72% of the group's profit, several flagged wallets traded only a few hundred dollars, and the top-ranked wallet (398 events, Z 12.5) earns 0.3% per dollar, which looks like a high-frequency or market-making account rather than an informed trader. (3) The empirical-null width is floored at 1.0, but the Z distribution is narrower (90th percentile 0.95 vs 1.28 for N(0,1)), so the test is conservative and likely under-flags. Not yet changed.

**Alert candidates.** `results_v3/alerts/candidates.parquet` (11.0M trades, 99,404 wallets) was built with `build_alert_candidates.py --data data_events_full/live --results results_v3` (about 1 minute; now streams parts to disk, low memory). It is NOT in the zip (about 400 MB; rebuild with that command). The alert evaluation on the full history has NOT been run yet. The fresh Oct-Nov 2026 window is inside the candidates; keep it out of design decisions (use `--test-end`, to be added) and use it once at the end.

**Not on the site/notebook yet:** results_v3 and the agreement features (section 15; promising, not proven). The site and notebook still read the 2026-only results_v2.

## 17. Forward windows and crowd-flow control (Nov 16)

Reason: the section 15 cut-offs had overlapping test periods (the 05-01 test contains the 06-01 and 07-01 periods), so 'positive in all three' was one stretch of time seen three ways, and agreement might only mean 'money is flowing into the favourite'.

**Code.** `alerts.add_context_features` now also adds `FLOW_FEATURES` (dollars from ALL qualifying wallets on the same and opposite side in the last 24 h, no skill filter). `run_alert_windows.py` trains before each window start and tests on trades made inside the window in events that resolved inside it (non-overlapping), pools all windows, and writes `results_v3/alerts_windows/` (`backtest_table.csv`, `pooled_table.csv`, `model_deciles.csv`, `run_info.json`). `evaluate_alerts.run_cutoff` gained `test_end`, `label`, `collect`. Models: all features, price only, crowd flow (no wallet skill), trade + agreement (no own history), all + agreement, all + agreement + flow. Default windows are quarterly from 2025-07-01 to 2026-10-01; **the fresh Oct-Nov 2026 window is not touched.** 48 tests pass. Smoke-tested only on a small slice (results meaningless); the full run is done on the Mac.

**How to read it.** Agreement is only interesting if 'All + agreement' (and 'Trade + agreement') beat 'Crowd flow' and 'Price only' in the pooled table with an interval that excludes zero, and in most windows. If crowd flow does as well, the signal is 'follow the money', not 'follow good wallets'.

## 18. Less conservative flagging rule (Nov 16): results_v4

**Why.** The original rule floors the empirical-null spread at 1 and centres Z at 0. On the full history the bulk of Z has a robust spread of only about 0.6 (the event variance formula is conservative: outcomes of one event are summed although their true covariance is negative), so almost nothing could be flagged (23 of 17,254 wallets). `skill_test.flag_wallets(..., sigma_floor=0, center=True)` estimates both spread and centre from the bulk (Efron empirical null). Defaults are unchanged; `run_flagging_v2.py --sigma-floor 0 --center` selects the new rule. `results_v4/` = full history with the new rule (`validation_report.json`, `likely_informed_wallets.csv`).

**Evidence (formation before the split, evaluation after; vs wallets matched on volume and event count).**

| rule | flagged (06-01 split) | out-of-sample mean z diff (CI) | pooled profit per dollar, flagged vs rest |
|---|---|---|---|
| R0 original | 23 | +0.28 (+0.09, +0.48) | 9.0% vs 1.8% |
| R1 spread free, centre 0 | 345 | +0.15 (+0.07, +0.23) | 5.7% vs 1.5% |
| R2 spread free + centre (adopted candidate) | 207 | +0.23 (+0.13, +0.34) | 6.7% vs 1.5% |

R2 across the three splits (05-01 / 06-01 / 07-01): flagged 194 / 207 / 215, z diff +0.11 / +0.23 / +0.27, CIs above zero in all (the 05-01 lower bound is +0.008). The EXTRA wallets only R2 flags (about 180 per split) also beat matched wallets out of sample: z +0.16 / +0.22 / +0.19 (all CIs above zero), pooled profit per dollar 6.4-7.0% vs 1.3-1.6%. Simulation (3 seeds): no false flags with pure noise under any rule; with 60 planted informed traders, recall rises from 30-53% (R0) to 77-88% (R2) at 96% precision. Placebo flag rate under R2: 0.56% vs 1.20% for events (R0: 0.5% vs 0.13%, i.e. the placebo flagged MORE than events; now it flags about half as many).

**Caveats.** (1) The rule was chosen after seeing out-of-sample results, and the three evaluation windows overlap, so treat this as promising until the fresh Oct-Nov 2026 window confirms it. (2) Placebo flagged wallets still show a positive out-of-sample z (+0.42, CI -0.06 to +0.95, 9 wallets), so persistence alone does not prove private information; the contrast with events is in the flag rate and in profit per dollar. (3) Concentration: the old R0 list (23 wallets) was dominated by one account (72% of group formation profit); in the new list (207 wallets) the top wallet has 13% and the top five 44%. But 30 flagged wallets traded under $1,000 in total and 34 earn under 1% per dollar (high-frequency style), so show them as 'likely informed' candidates, not proven insiders.

## 19. Forward windows result (Nov 16): agreement does NOT hold up; the simple wallet-record rule does, modestly

`python run_alert_windows.py` on the full history: five non-overlapping quarters (2025-07 to 2026-09), train before each window, test inside it, $100 flat per alert, 5 min delay (results in `results_v3/alerts_windows/`).

Pooled over windows (5 min delay; random/all candidate trades -3.7%):

| strategy | alerts | events | ROI (95% CI) | windows positive | top wallet share |
|---|---|---|---|---|---|
| Old alert model (all features) | 4,383 | 223 | -20.7% (-42.0, +0.7) | 3/5 | n/a |
| Crowd flow, no wallet skill | 13,413 | 255 | -2.3% (-25.7, +21.9) | 2/5 | n/a |
| Trade + agreement | 12,439 | 172 | +8.9% (-16.4, +33.0) | 2/4 | 0.7% |
| All + agreement + flow | 11,738 | 170 | +12.9% (-15.1, +46.9) | 3/5 | 0.5% |
| Trade with flagged wallets | 3,147 | 588 | +7.8% (-0.3, +18.4) | 4/5 | 56% |
| **Past score Z >= 3, 10+ events (no model)** | 7,023 | 643 | **+8.1% (+1.6, +14.6)** | 3/5 | 24% |

Per window, the rule: -10.7%, +15.2%, +18.2%, -1.2%, +15.3%; with 1 cent extra cost: -15%, +7%, +12%, -4%, +10%; at 0 min delay pooled +13.7% (+6.5, +21.4); at 1 hour +7.1% (+1.0, +14.4). The Z >= 3 / 10-events threshold was fixed before these windows and not tuned on them.

Reading. (1) The model-based alerts are unstable: alerts per window range from 0 to 6,400 (the 99th-percentile threshold from the validation slice does not carry over), the first window loses 99.8% on a handful of events, and 2026 Q2-Q3 supply about 95% of agreement alerts. The earlier +16% (cut-off 06-01) came from the Q2 window; it does not generalise. The old alert model on the site (all features) is negative pooled. (2) Agreement beats the crowd-flow control by about 11 points but the interval is far too wide to claim an effect. (3) The simple rule 'trade with wallets whose past record is strong' is the only strategy with a pooled interval above zero, at about +8% per dollar before real-world costs (liquidity, fees beyond the 1 cent test), roughly 12 points better than random trades.

Consequences. The site's Alerts tab (model based) needs to be reworked around the rule and the negative model result should be stated plainly. Not yet done. The fresh Oct-Nov 2026 window has still not been used; run the frozen rule on it once.

## 20. One-shot fresh-window test (Nov 16): the frozen rule on Oct 1 to Nov 14, 2026

`python run_alert_fresh.py` (writes `results_v3/alerts_fresh/`). 1.2 million candidate trades, 256 events, none used for any earlier decision. Nothing is trained; the rule and its thresholds were fixed before any of the window tests (section 19). $100 per alert, price of the next trade after the delay, held to resolution.

| strategy (5 min delay) | alerts | events | wallets | return (95% CI) | random-draw mean |
|---|---|---|---|---|---|
| **PRIMARY: past score Z >= 3, 10+ events** | 3,507 | 238 | 60 | **+8.4% (-0.8, +18.0)** | +4.0% |
| flagged wallets, original rule (21 wallets) | 2,346 | 238 | 21 | +10.4% (-0.2, +20.9) | +4.0% |
| flagged wallets, less conservative rule (108 wallets) | 5,831 | 241 | 108 | +8.5% (-2.3, +20.7) | +4.0% |
| all candidate trades | 98,842 | 252 | 9,968 | +4.1% (-6.2, +15.5) | |

Other delays for the primary rule: 0 min +9.5% (+1.3, +18.8); 1 hour +6.2% (-3.4, +16.7). With 1 cent extra cost per share: 5 min +4.8% (-3.6, +13.5) against random +0.7%; 1 hour +2.7% (-5.9, +12.1). Top wallet's share of the primary rule's profit: 17%.

Reading. The point estimate (+8.4%) matches the five earlier windows (+8.1% pooled), so the result did not fall apart on unseen data. But this window alone does not reach significance (the interval includes zero), and random candidate trades also did well here (+4%), so the edge over random is about 4 points, within noise. Across the five forward windows plus this one the rule was positive in 4 of 6 (-10.7%, +15.2%, +18.2%, -1.2%, +15.3%, +8.4%). A fair summary: a modest positive edge that is consistent in size but not statistically established, and it shrinks quickly with trading costs.

## 21. When do likely-informed wallets trade? (timing analysis, Nov 16)

`python timing_analysis.py process` then `python timing_analysis.py report` (about 15 seconds each; code in `timing_analysis.py` and `make_timing_figures.py`; tests in `tests/test_timing.py`). Outputs in `results_v4/timing/` (`timing_clock_shares.csv`, `timing_markouts.csv`, `timing_by_shift_bucket.csv`, `timing_fomc.csv`, `timing_summary.json`, `figures/`).

**Design.** The 207 wallets flagged BEFORE 2026-06-01 (less conservative rule) are compared with the 17,000 other experienced wallets and with all traders, using only events that resolved AFTER 2026-06-01 (1,309 markets, 5.25 million trades), so wallets were never selected for the timing being measured. Four clocks: share of the market's life elapsed; hours before resolution; hours relative to the market's biggest hourly price shift (528 markets with a shift of at least 15 cents); and minutes relative to its sharpest 10-minute jump (631 markets with a jump of at least 10 cents). Plus markouts (price move in the trade's favour afterwards), and the FOMC statement time for Fed markets. Each wallet counts once in the shares (large wallets do not dominate); intervals resample wallets.

**Findings (flagged vs other experienced wallets, share of each group's dollars).**
- **Earlier, not last-minute.** First 10% of a market's life: 5.5% (CI 3.8 to 7.5) vs 3.7%; 90+ days before resolution: 29.6% (23.7 to 35.6) vs 24.6%. No excess in the final hours: under 1 hour before resolution 0.2% vs 0.3%; 1 to 6 hours 5.9% vs 4.2%. (All traders pooled put 51% of dollars in the last 10% of life because big dollars arrive late.)
- **Bursts around sharp price jumps.** The 10 minutes BEFORE the sharpest jump: 2.0% of flagged dollars (CI 1.0 to 3.0) vs 0.6% (3.3 times as much); 69.5% of those dollars were on the side the price then moved to (CI 57.6 to 78.6) vs 47% for others (a coin flip); profit +57% per dollar (CI +24 to +98) vs about 0% for others. 10 to 30 minutes before: 1.5% vs 0.5%, 60% vs 44% on the right side. Robustness: the top wallet is 16% of that bucket's dollars; without the top five wallets the right-side share is 58% and the profit +74% per dollar.
- **After the jump there is no edge.** In the 10 minutes of the jump flagged wallets are 3.7% of dollars (vs 1.2%) and 75% on the right side (others 66%), but profit is about 0; 30 to 60 minutes after, profit is -32% (CI -56 to 0). Fast reaction, not anticipation.
- **Markouts.** After their trades the price moves their way: +1.7% per dollar within 5 minutes (CI +1.0 to +2.7), +2.1% in 1 hour, +4.5% in 6 hours, +8.8% in 24 hours (CI +2.8 to +19.4); at resolution +8.0% (CI -3.8 to +26.5). Other experienced wallets: +0.2%, +0.1%, +0.7%, +0.8%, and -4.7% at resolution.
- **Fed statements.** 88 flagged wallets traded Fed markets in the period: 81% of their dollars were placed more than 24 hours before the statement, 0.9% in the 15 minutes before (others 1.0%), but 6.0% within an hour AFTER it (others 0.7%): quick reaction, no visible pre-statement trading.

**Caveats (important).** (1) The signal before sharp jumps is concentrated: in only 48% of the 332 markets where flagged wallets traded in that window did most of their dollars land on the right side (54% of wallets, vs 42% of other experienced wallets), so it comes from a subset of large, well-timed trades, not every market. (2) A large buyer who starts pushing the price a few minutes before the jump bin makes the jump follow its own trades, so price impact and information cannot be separated. (3) A positive 5-minute markout can also come from patient limit orders or market making. (4) The flagging rule was chosen after seeing results (section 18). (5) Single-bucket profit figures are volatile (long shots); use the share on the right side and the intervals. (6) Placebo wallets have not been run through this analysis yet.
