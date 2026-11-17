# PolyInsiders — ECE 473: pre-resolution wallet and event prediction
**Aiden Sandler & Emilly Antero | Updated November 16, 2026**

## Start here
This is a **working local research prototype**, designed to match the provided PolyInsiders `findings.html` aesthetic. It contains *two distinct* forward-looking models and an auditable tracker. The trained models in `data/run/` were fit on our full historical Polymarket archive (see `data/run/model.json`). `python -m polyinsider.cli demo` regenerates a separate, clearly labeled **synthetic** demo for testing the pipeline only. Historical results are from the team's own runs; see `analysis/` and the top-level README for the latest tests.

**Real-data status:** the 15.6-million-fill archive is not in the repository (too large; see `analysis/README.md`). Trained models, derived results and a saved live snapshot are included. Live collection requires an internet connection and `serve --live`; the hosted website is a static snapshot.

## One-command demo (after dependencies)

```bash
cd polyInsider   # the cloned repository folder
python3 -m venv .venv
source .venv/bin/activate                 # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt   # includes duckdb for the timing builders
python -m pytest -q                     # 16 automated tests
python -m polyinsider.cli serve --live --interval 120 --port 8000   # real snapshots; needs internet
# python -m polyinsider.cli demo   # OPTIONAL: overwrites with a synthetic demo; do not run if you want to keep the real models
```
Browse `http://127.0.0.1:8000`. The **Tracker** page shows the saved snapshot (labeled synthetic only if you ran `demo`). Refreshing the browser reloads saved JSON; it does not poll Polymarket by itself.

## The TA's requirement and our two prediction targets

| Model | Prediction frozen before resolution | Target for later validation | Does it identify people who possess private information? |
|---|---|---|---|
| **Wallet model** (`train-wallets`) | Uses only wallet event-level signed-price advantages from fully resolved *earlier* events | Whether the wallet has positive average signed price advantage across **later** events in which it trades | **No.** It measures forecastable future outperformance, not actual private information |
| **Market model** (`train`) | Uses contemporaneous market YES price, recent signed flow, historically estimated wallet skill, number of wallets and observed volume | Whether **YES** eventually resolves | **No.** It predicts outcomes, not insider status |

The original team's more selective multiple-testing score is a **third** research artifact. Its previously reported results should not be presented as results for the new supervised wallet model without rerunning on the archived data.

### What “informed” means here
An empirical operational definition can be “wallets with unusually strong earlier trading outcomes and elevated predicted probability of outperforming public prices again.” We cannot observe ground-truth possession of information. Do not train a supposed **insider/non-insider** supervised classifier using prior winning status as if it were true insider labels.

## Historical data: three ways to get started

**Option A: original prepared history (most direct).** Use a genuine `historical_snapshots.csv` with columns:

`event_id,condition_id,title,snapshot_time,resolution_time,yes_price,wallet,direction,amount_usd,resolved_yes`

Each market must be frozen at a timestamp preceding real resolution; `yes_price` must be the then-current public price, not today's value. `wallet` is a stable public proxy identifier. `direction` is YES or NO observed exposure. `resolved_yes` is a verified 0/1 label used **only** after resolution when training historical examples.

**Option B: original 15.6M-fill archive + verified market metadata.** Use `prepare --stream` to avoid loading the entire archive into memory. Example market metadata format is in `data/templates/verified_market_metadata_TEMPLATE.csv` (its row is a placeholder, **not** real evidence). Required: a real, independently verified outcome and **actual** resolution timestamp for each binary market; a snapshot freeze timestamp chosen without peeking at the outcome.

```bash
python -m polyinsider.cli prepare \
  --trades /absolute/path/original_v2_fills.csv \
  --markets /absolute/path/verified_market_metadata.csv \
  --stream --out data/prepared_history.csv
```

**Option C: download public historical taker fills from the v2 API for a verified list of market IDs.** This requires the **internet on your own machine** and may take time or fail on API limits. The downloaded file is not a substitute for independently verified `resolved_yes`, `resolution_time` or `snapshot_time`.

```bash
python -m polyinsider.cli download-trades \
  --markets /absolute/path/verified_market_metadata.csv \
  --out data/downloaded_v2_taker_fills.csv
# Inspect `data/downloaded_v2_taker_fills.download_report.json`.
# Stop if it lists truncation or failed markets.
python -m polyinsider.cli prepare \
  --trades data/downloaded_v2_taker_fills.csv \
  --markets /absolute/path/verified_market_metadata.csv \
  --stream --out data/prepared_history.csv
```

**Important:** The `prepare` utility produces **signed recent trade-flow proxies**, not true positions, realized round-trip ROI, full order books, or audited PnL. Research claims must match this distinction. Trades need full coverage in each pre-snapshot 48-hour window and taker-only provenance; otherwise the price/flow features may be biased. Do not blindly use truncated downloads or Gamma's current prices for historical snapshots.

## Train on REAL verified history

```bash
python -m polyinsider.cli train \
  --csv data/prepared_history.csv --source 'REAL VERIFIED PRE-RESOLUTION SNAPSHOTS' \
  --out data/run
python -m polyinsider.cli train-wallets \
  --csv data/prepared_history.csv --source 'REAL VERIFIED PRE-RESOLUTION SNAPSHOTS' \
  --out data/run
```

The `train` command saves `model.json`, `wallet_state.json`, `backtest.json`, and `asof_features.csv`. Market-model holdout events begin only after the last training event resolves. The `train-wallets` command saves `wallet_model.json`, `wallet_probabilities.csv` and `wallet_backtest.json`. It uses two early chronological training windows and one untouched later wallet evaluation window. Training labels require at least three later events per wallet; the final model is then refitted on all already-resolved training examples for future use. Review samples and AUC/Brier metrics before reporting them.

To prevent accidentally using events resolved after a specific time, include `--cutoff 2026-10-01T00:00:00Z` for market model or `--asof 2026-10-01T00:00:00Z` for wallet model. For genuine future polling, use the latest complete verified history **prior to capture**. These scripts do not run in a hosted background service.

## Poll and evaluate unresolved real markets

```bash
# Requires internet and a REAL-trained model in data/run:
python -m polyinsider.cli snapshot --limit 8
python -m polyinsider.cli serve --live --interval 120 --port 8000
# Press Ctrl+C to stop. The collector runs only while the local process runs.
```

The collector requests open markets from public Gamma, and taker trades from the public Data API v2. It filters fills after the snapshot time and older than 48 hours, paginates at most four pages per market, and declines to score if fewer than three eligible rows remain. Those coverage limits MUST be described. Every saved market prediction is appended to `data/run/predictions.csv` and additionally to a SHA-256 hash-chained `data/run/predictions_audit.jsonl`.

```bash
python -m polyinsider.cli verify-log
python -m polyinsider.cli evaluate \
  --predictions data/run/predictions.csv \
  --resolutions /absolute/path/verified_resolutions.csv
```

Evaluation takes the **first** saved forecast per market, checks capture was before *actual verified* resolution, and compares Brier, log loss and ROC AUC with the contemporaneous market YES price. An empty evaluation means **we cannot yet validate a prospective prediction**, not that it worked. The supplied template has example placeholders, not trustworthy resolution data.

### Debug live public API with synthetic model (DO NOT report as empirical result)

```bash
python -m polyinsider.cli snapshot --allow-demo-model
# Mode should be: LIVE API + SYNTHETIC MODEL — NOT VALIDATED
```

## What the team's prior research reported (not newly recomputed)
- Scheduled-event data: 1,610 markets/outcomes, 15.6M fills, 548,066 wallets (Jan–Sep 2026); 300 crypto up/down comparison markets.
- Later recalibrated skill test: 19 flagged of 4,009 eligible event wallets; rank correlation 0.24 (CI 0.20–0.28) on scheduled events versus 0.07 (0.02–0.12) in the crypto comparison; AUC 0.65 versus 0.55.
- Crucial caveat: top-scoring wallets had approximately the **same later edge +0.13** in both groups, so stronger correlations do not prove insider information.
- Findings were described in the prior team's `handoff.md` and November 11 documents and were not independently reproduced here. Historical wallet-score metrics and the new synthetic model metrics are **not interchangeable**.

## Synthetic pipeline tests (not real-data evidence)
The included deterministic generator creates 220 artificial markets, intentionally endows some artificial wallets with information about simulated outcomes, and produces 7 fake currently unresolved market examples. These are simulated for pipeline testing. Model statistics can be viewed on the tracker and in `data/run/*backtest.json`. A high synthetic AUC is not evidence of useful real forecasts.

## Ethical, empirical, and data limitations
- No ground-truth labels for access to private information; no allegation about any real wallet.
- Public wallet identity is not guaranteed to represent one person; account linking is outside scope.
- Historical market data may omit negative-risk conversions, partial positions, fees, or heterogeneous settlement.
- Market feature `yes_price` in historical prepared data is the most recent fill proxy, while live Gamma gives current outcome prices, a source of train–serve mismatch.
- Current taker-only feeds are limited lookback samples, not the full book or positions.
- Repeated wallet observations across chronological training windows mean reported wallet-level AUC does not itself justify a conventional independent-wallet confidence interval.
- The local hash-chain guards against unintended file modifications but is not digitally signed or third-party timestamped.
- A genuine “live tracker” requires running the collector on an internet-connected host and actually observing markets before they resolve.

## Official public documentation
- Polymarket Data API v2: https://data-api.polymarket.com/v2/docs
- Gamma API: https://docs.polymarket.com/developers/gamma-markets-api/overview
- Public researcher overview: https://institute.polymarket.com/data

## Reproducibility and ownership
`polyinsider/`: source; `tests/`: 16 automated tests; `web/`: interactive local website; `data/`: trained models, saved results and input templates; `docs/`: detailed explanation and TA walkthrough; `slides/`: editable presentation. All source files are provided, no API keys or private wallet secrets required.
