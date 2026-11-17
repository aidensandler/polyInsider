# ECE 473 - Polymarket likely-informed wallets (current iteration: Nov 16, 2026)

**Start with `handoff.md` ("READ FIRST" block at the top).** The full-history work of Nov 16 is described there and in steps 9 to 11 below.

**Current workflow** (run from ~/pm473, any order that respects the arrows):

```bash
# 1. data (already done; only re-run to add markets)       run_milestone1.py ... --stage download / analyze
# 2. flag wallets with the calibrated test + validation   python run_flagging_v2.py --data data_events/live --placebo data_placebo/live
# 3. performance statistics                                python evaluate_model.py
# 4. figures, tables, captions for the writeup             python make_report_figures.py      -> results_v2/report/
# 5. live labels, then the website                         python run_flagging_v2.py --data data_events/live --label-all
#                                                          python build_dashboard.py --data data_events/live --results results_v2
# 6. live feed (own terminal) and the site (own terminal)  python live_feed.py        |   python -m http.server -d site 8000
# 7. alert model (Nov 13): features, held-out test, copy-trading backtest, figures, live scorer
#                                                          python build_alert_candidates.py            # about 15 s
#                                                          python evaluate_alerts.py                   # main cut-off; add --cutoff 2026-05-01 / 2026-07-01, then --merge
#                                                          python make_alert_figures.py                # -> results_v2/alerts/figures/
#                                                          python evaluate_alerts.py --live-model ; python live_alerts.py
# 8. the walkthrough notebook (steps 1 to 16 on 2026 data, Part B steps 17 to 21 on the full history)               jupyter notebook polyinsider_walkthrough.ipynb
# 9. more data and the full history (Nov 16)
#    download windows:   python run_milestone1.py --out data_events_2025 --trades-dir data/live/trades_raw --end-min 2025-01-01 --end-max 2025-12-31 --n-markets 6000 --stage download --tags fed fed-rates ... (see handoff.md 14)
#    combine + analyse:  python combine_market_lists.py data_events/live data_events_2025/live data_events_fresh/live --out data_events_full_markets.parquet
#                        python run_milestone1.py --out data_events_full --trades-dir data/live/trades_raw --market-list data_events_full_markets.parquet --n-markets 100000 --end-min 2025-01-01 --end-max 2026-11-15 --audit-onchain 0 --stage analyze
#    flag wallets:       python run_flagging_v2.py --data data_events_full/live --placebo data_placebo/live --out results_v3 --split-date 2026-06-01 --skip-synthetic
#                        python run_flagging_v2.py ... --out results_v4 --sigma-floor 0 --center        # less conservative rule (handoff 18)
# 10. alert tests on the full history (about 20 GB of memory; run on the Mac)
#                        python build_alert_candidates.py --data data_events_full/live --results results_v3
#                        python run_alert_windows.py          # five forward quarters + pooled table (handoff 19)
#                        python run_alert_fresh.py            # one-shot test of the frozen rule on Oct-Nov 2026 (handoff 20)
# 11. when do informed wallets trade? (handoff 21)       python timing_analysis.py process ; python timing_analysis.py report     # -> results_v4/timing/
# 11. optional compute: cloud_run.sh (rented VM) or adroit_analyze.slurm (Princeton Adroit); both untested
python -m pytest -q tests                                  # all tests (54)
```

Current files: `pm473/skill_test.py`, `model_eval.py`, `trade_returns.py` (new), `run_flagging_v2.py`,
`evaluate_model.py`, `make_report_figures.py`, `build_dashboard.py`, `live_feed.py`, `pm473/alerts.py`, `pm473/alert_backtest.py`,
`build_alert_candidates.py`, `evaluate_alerts.py`, `make_alert_figures.py`, `live_alerts.py`, `run_alert_windows.py`, `run_alert_fresh.py`, `timing_analysis.py`, `make_timing_figures.py`, `combine_market_lists.py`, `site/`, `results_v2/` (2026 only), `results_v3/` and `results_v4/` (full history),
`handoff.md`. Anything from before Nov 11 that is no longer used is in `archive_before_nov11/`
(see its README). The older sections below describe the Milestone I pipeline, which the new code still builds on;
commands for `run_robustness.py`, `cleanup_raw.py` and `smoke_test.py` below now live in the archive.

---

# ECE 473 — Milestone I pipeline (Polymarket informed-trader analysis)

Collect → clean → PnL → behavioural features → informed/uninformed labels → out-of-sample comparison to a baseline.

## Commands for the two samples

Main sample (download first, analyze when your computer is free):

```bash
caffeinate -i python run_milestone1.py --out data_events --trades-dir data/live/trades_raw --n-markets 800 --audit-onchain 200 --stage download
python run_milestone1.py --out data_events --trades-dir data/live/trades_raw --n-markets 800 --audit-onchain 200 --stage analyze
```

Crypto placebo. This selects from the 800-market random sample your first run saved in `data/live/markets.parquet` (no re-listing) and reuses crypto markets already downloaded:

```bash
caffeinate -i python run_milestone1.py --out data_placebo --trades-dir data/live/trades_raw --market-list data/live/markets.parquet --n-markets 300 --tags --include-slug-pattern "up-or-down|updown" --exclude-slug-pattern "" --min-life-days 0 --exclude-neg-risk --prefer-downloaded --stage download
```

Then rerun with `--stage analyze`.

Robustness checks (reads saved outputs; the duplicate check needs `POLYGON_RPC_URL`):

```bash
python run_robustness.py --data data_events/live --placebo data_placebo/live --check-dupes 50 --trades-dir data/live/trades_raw
```

Free disk space taken by duplicate raw trade pages (keeps the Gamma listings):

```bash
python cleanup_raw.py --dry-run data/live/raw data_events/live/raw
```

Both use the default dates (Jan 1–Sep 30, 2026) and split date (June 1). Each stage prints its peak memory use.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest -q tests                 # PnL accounting + on-chain decoder
python run_milestone1.py --synthetic       # full pipeline offline, with planted informed traders
python smoke_test.py                       # ~10 live requests; confirms real field names
python run_milestone1.py --end-min 2026-01-01 --end-max 2026-09-30 \
       --split-date 2026-06-01 --min-volume 50000 --n-markets 800
# optional on-chain audit (needs POLYGON_RPC_URL or ETHERSCAN_API_KEY):
python run_milestone1.py ... --audit-onchain 200
```

Outputs land in `data/live/`: `markets.parquet`, `trades_raw/` (one file per market, so the run can resume), `wallet_market_pnl.parquet`, `wallet_panel.csv`, `milestone1_report.json`, and a figure. Every raw API page is also saved to `data/live/raw/`, so the dataset can be reproduced.

## Data in this bundle (recovered October 28)

- `data/live/markets.parquet`: the 800-market random sample from the broad run (all 118,809 eligible binary markets). It includes 417 crypto up/down markets and is the placebo's source.
- `data/live/trades_raw/`: 272 trade files, 244 from that sample and 28 from the very first run. Event-sample downloads also go here.
- `data/live/raw/gamma_listing_compact.json`: the full 141,278-market Gamma listing, compacted to the fields the code uses. Usable with `--gamma-cache data/live/raw`.
- `data/live/first_run_28_markets.parquet`: the first run's market list, kept for completeness.

The event sample (`data_events/`, 1,610 markets) was not recoverable and has to be downloaded again.

**Updating code without touching data.** Code updates come as a zip of code only. Unzip them with Terminal, from the folder that contains `pm473`:

```bash
unzip -o pm473_milestone1.zip
```

Never use Finder's "Replace" on the `pm473` folder. It deletes everything inside, including data.

## Research design: event markets plus a crypto placebo

The main sample is scheduled-event politics and economics markets: Fed decisions, economic data releases, and elections. In these markets, private analysis plausibly exists and the deciding information becomes public at a known time. Recurring crypto "up or down" markets form a separate placebo sample. Their outcome depends on a public price feed, so a method that flags many "informed" traders there is picking up speed or noise rather than private information.

Each market gets an `info_time`, recording when its deciding information became public, and `info_time_source` records how that was set:

- `fomc_statement`: the official 2:00 pm ET FOMC statement time (2026 dates in `config.py`), for US Fed markets.
- `event_start`: Gamma's event start time, for games and debates.
- `scheduled_end`: the market's scheduled end date otherwise. For elections this is roughly when polls close; results arrive after.

Every fill gets `hours_to_info`, and each wallet gets the share of its volume in the 24 hours and 7 days before `info_time`, its share after `info_time`, and its median hours before `info_time`. These are the delayed-revelation measures from Chen et al.

Multi-outcome ("neg-risk") events are now included by default. Each outcome is its own YES/NO market, so PnL is computed per outcome. One caveat: conversions between outcomes are not in the trade feed, so a wallet that converts may show a split adjustment it didn't make. `pnl_sanity` in the report shows how often split adjustments occur.

Selection defaults:

- **Tags:** the topic tags in `collect.DEFAULT_TAGS`. Unknown slugs are skipped with a warning. Run `python list_tags.py fed` (or `election`, `econ`) to see what exists, then pass your choice with `--tags`.
- **Recurring crypto:** dropped by slug pattern.
- **Short markets:** markets open less than 2 days are dropped (`--min-life-days`), since there is no room to delay trading in them.

## API changes since the proposal was written

Several things changed in October 2026, and these choices follow from them.

- **Data API v1 was retired on Oct 24.** All trade reads use `/v2/trades`, which has cursor pagination, snake_case fields, and lowercase `true`/`false` parameters.
- **CLOB `/prices-history` shuts off on Oct 30.** Its replacement, `data-api /v2/prices-history`, keeps 1-minute prices for only about 7 days, and older history comes in 3-hour and 12-hour buckets. For that reason, markouts are computed from the trade tape itself rather than from price history.
- **No official historical order books exist.** `/book` returns only the live book, so historical spreads and depth come from PolymarketData instead. See the order book section below.
- **Exchange contracts changed on Apr 28, 2026 (CLOB V2).** There are new addresses and a new `OrderFilled` layout. `onchain.py` decodes both layouts. Fills carry a `post_v2_exchange` flag so you can control for the regime change.
- **Protocol V2 markets start around Nov 2.** These use PositionManager and ExchangeV3, with new position IDs. The sample is restricted to Gamma `version == "v1"` markets.
- **PolygonScan's API is now Etherscan API V2** (`chainid=137`). Etherscan has removed some chains from its free tier. If you get "Free API access is not supported for this chain", use a free RPC endpoint from Alchemy, Infura, or QuickNode via `POLYGON_RPC_URL`.

## Method choices, and the bias each one avoids

- **Market sample.** Markets are drawn at random from all eligible resolved binary markets, not ranked by volume. A volume-ranked sample over-weights headline markets. Every exclusion is counted in `report["selection"]`.
- **On-chain as ground truth, cheaply.** Instead of indexing every log, `--audit-onchain N` checks a random sample of Data API fills against `OrderFilled` events in their transaction receipts. A match rate near 100% justifies using the Data API tape. This also responds directly to Dubach's 59% finding, which concerned the order-book feed rather than settlement data.
- **PnL from trades alone is exact for binary markets.** If a wallet sells shares it never bought, those shares must have come from a split, so the code adds $1 complete sets. Merges and redemptions are captured by valuing the final inventory at the resolution payout (see the `pnl.py` docstring and the tests). Neg-risk markets are excluded by default because cross-market conversions break this identity.
- **Order splitting excludes sweeps.** One taker order filling against several makers produces several rows in the same transaction. Fills are collapsed to one row per (wallet, market, transaction, direction) before runs of child orders are measured.
- **No circular labels.** Wallets are labeled using only markets that resolve before `--split-date` (formation). The labels are then judged on markets that resolve after it (evaluation). Comparing formation-period PnL would be circular, because wallets were selected for having high PnL in that period.
- **Multiple testing.** A plain t > 2 cutoff across thousands of wallets flags about 2% of pure-noise traders. The code applies a Benjamini–Hochberg correction (q = 0.10) to one-sided per-market ROI t-tests, using markets rather than fills as the unit of observation.
- **Two baselines.** Informed wallets' evaluation ROI is compared against all other wallets and against a volume-matched random sample, with bootstrap 95% confidence intervals. The ML models report cross-validated AUC against volume-only and past-ROI-only baselines.

## What the synthetic run shows (and what it means for the real pull)

The synthetic world has noise traders, market makers, and informed wallets that are right 80% of the time and split their orders.

| markets | precision | recall | noise false positives |
|---|---|---|---|
| 160 | 1.00 | 0.04 | 0 |
| 400 | 0.94 | 0.34 | 1 |
| 800 | 1.00 | 0.86 | 0 |

Each binary resolution is one very noisy draw. Even a genuinely skilled wallet needs many markets before its skill is statistically distinguishable from luck. Pull at least 800 markets, and treat "not flagged" as "not proven informed" rather than "uninformed". The first version of the generator had informed wallets trade after the price had already moved, and they earned almost nothing. In the real data, expect the informed signal to appear in the trades placed before a price move, which is the part of your hypothesis about delayed revelation.

## Historical order books (spreads, depth, Kyle's lambda)

Polymarket's API has no past order books, so two sources are wired in.

**PolymarketData (main source, for your Jan–Sep 2026 sample).** It has 1-minute L2 snapshots for every category since August 2025, looked up by Gamma's market slug. Get a free key at app.polymarketdata.co, then:

```bash
export POLYMARKETDATA_API_KEY=...
python run_books.py --probe                     # 1 market, 1 hour; check the parser finds books
python run_books.py --n-markets 100 --days 7    # last week before resolution, 100 random markets
```

Run `--probe` first. I couldn't test against their live API, so if it prints "parser found 0 snapshots", send me the printed JSON. The free tier's limits aren't published, so start small. If you hit a cap, lower `--n-markets`, reduce `--days`, or use `--resolution 10m`.

The outputs are:

- `book_minutes.parquet`: mid, spread, depth within 1¢ and 5¢, and imbalance, per market-minute, in outcome-0 units.
- `fills_with_book.parquet`: every fill with the book it hit, the half-spread paid, its size relative to depth, and whether it traded with the book's imbalance.
- `market_kyle_lambda.csv`: per-market OLS slope of mid-price change on signed order flow (Kyle 1985), per $1k of flow.
- `wallet_panel_with_books.csv`: the Milestone I panel plus execution features.

**Your own recorder (free, going forward).** `record_books.py` saves Polymarket's live WebSocket book feed to `data/ws/` as hourly gzipped files. Start it now on a machine that stays on, and you'll have tick-level books for markets resolving during Milestone II:

```bash
python record_books.py --min-volume 50000 --max-markets 300
```

## Files

- `pm473/config.py`: endpoints, contract addresses, cutover dates
- `pm473/api.py`: rate-limited client with retries, v2 cursor paging, and raw-page archiving
- `pm473/collect.py`: market selection (Gamma) and trade download (Data API v2)
- `pm473/clean.py`: normalization with an audit log of every dropped row
- `pm473/onchain.py`: V1/V2 `OrderFilled` decoder and sample audit
- `pm473/pnl.py`: wallet × market PnL
- `pm473/features.py`: markouts (5m/1h/24h/terminal), timing, order-splitting features
- `pm473/classify.py`: skill tests with FDR, formation/evaluation split, baselines, ML
- `pm473/books.py`: PolymarketData client, defensive parser, book features, fill join, Kyle lambda
- `run_books.py`, `record_books.py`: historical book step and live recorder
- `run_robustness.py`: matched comparisons, coefficient CIs, GB diagnostics, duplicate and placebo checks
- `cleanup_raw.py`: deletes redundant archived trade pages
- `pm473/synthetic.py`: generator with planted informed traders

## Toward Milestone II

`wallet_panel.csv` is already shaped for the next step. Strategy classification can cluster the timing, splitting, and markout features of flagged wallets. The game-theory mapping can test whether informed wallets' trading is back-loaded relative to when their information arrives, which is the Chen et al. prediction.
