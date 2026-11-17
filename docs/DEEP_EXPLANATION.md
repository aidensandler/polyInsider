# PolyInsiders — deep explanation and how to present it

**Version:** Final consolidated technical explanation, November 16, 2026. **Evidence:** the user's terminal reports plus the project code. The real Parquet archive and live Mac-trained models are not included in the shared download; therefore the numeric results here have not been re-estimated in this environment.

## 1. What were we trying to learn?

The original research investigated whether some Polymarket wallets did consistently better than chance or the public market price. A winning wallet is not necessarily informed about private facts: chance, speed, modeling skill, automation, or different risk exposure could all lead to good performance. Earlier team analysis generated historical evidence scores and examined whether wallet ranking persisted into later events.

The TA added a requirement: show a system that can compute and save predictions **while an event is unresolved**, using a model whose coefficients were learned from older resolved events. We met the engineering requirement through two models plus a local collector. But an engineering demonstration and proof of forecasting improvement are separate accomplishments.

## 2. What did Aiden's original data contain?

The user verified that Aiden's `data_events/live/fills_clean.parquet` contains **15,581,978 executed trades**, each with time, price, size, side, wallet and market integer codes; `markets_coded.parquet` contains 1,610 market identities and outcomes; `wallet_codes.parquet` maps wallet integer codes back to public addresses. `wallet_market_pnl.parquet` contains market-level wallet profit and loss. We did not need to download historical trades again.

Our converter wrote **164,129 prepared rows** from **1,175 binary-market snapshots** involving **491 events** and **82,586 wallets**. Each row includes `event_id`, `condition_id`, `title`, `snapshot_time`, `resolution_time`, `yes_price`, `wallet`, `direction`, `amount_usd`, and `resolved_yes`. There were no missing values in the prepared file.

**Important:** This 164,129-row file is not a record of 164,129 independent resolved markets. Multiple wallet rows refer to one market snapshot; related markets belong to the same event. The 24-hour nominal lead and 48-hour lookback define observation windows but do not alone certify that public information was not released already.

## 3. What did we mean by an “as-of” feature?

A feature is “as of” a particular snapshot only if its inputs were public before or at that snapshot. The model must not use a market's eventual YES/NO resolution, later price updates, future wallet trades, or other outcomes within the same multi-market event to compute a feature at an earlier time.

The `yes_price` input is the market's contemporary public probability proxy. Other features are recent net YES-versus-NO taker flow, number of distinct observed wallets, logarithm of traded volume, and `skill_flow` (the amount-weighted signed trade flow from wallets with historical evidence of skill). A wallet's score must only use events fully resolved before that snapshot. **The final empirical timing audit still needs independent release timestamps.**

## 4. The two trained models

### (A) Market outcome prediction

A regularized logistic regression predicts whether YES will eventually resolve. Its probability output is compared with the contemporaneous YES market price. Training features are from a snapshot before the observed resolution. In a single chronological evaluation, 97 markets (31 events) were used for training and 303 markets (148 events) for testing, with no train-event resolution after the earliest test snapshot. A separate **full-model fit** on all 1,175 eligible markets was used for later live tracking. Those 1,175 are *not* 1,175 held-out test cases.

Actual terminal output: AUC **0.9008**, Brier **0.1276**, log loss **0.4065**; the raw market price had AUC **0.9015**, Brier **0.1244**, log loss **0.3908**. The model did *not* beat the public price benchmark.

### (B) Wallet future-outperformance prediction

The supervised wallet model predicts whether a wallet will show positive average signed price advantage over market prices on at least three later resolved events, based on historical wallet performance indicators from prior completed events. It cannot be a supervised “insider/not insider” model because there are no verified private-information labels. On **504 held-out wallets**, the model reached **AUC 0.5109** and Brier **0.2519**, worse than the constant-base-rate Brier **0.2510**. This is near chance discrimination and not a reliable wallet classifier in the current specification.

## 5. Did the wallet-skill feature help outcome forecasting?

We trained the market model twice on exactly the same training/test split, once with `skill_flow` and once without. On the **303** held-out markets, **300** had nonzero skill-flow. The full model had Brier **0.1276** and AUC **0.9008**, while the reduced model had Brier **0.1275** and AUC **0.9009**. Full-minus-reduced Brier difference = **+0.000026** (event-bootstrap 95% CI **[-0.000057, +0.000119]**). AUC difference = **-0.000092** (95% CI **[-0.000533, +0.000267]**).

Both intervals include zero, so these data do not establish a detectable incremental benefit from the feature. This is more precise than saying “wallet skill never helps”: another skill definition, a different sample or later market segment might give different results. We have measured one specific specification.

## 6. What was live on the website?

The website at `http://localhost:8000/tracker.html` displayed a real Polymarket public API snapshot made on **November 16, 2026**. According to user-provided output, the `mode` said **LIVE PUBLIC API — REAL MODEL**. It tracked **8** markets, with **144** saved snapshots between **22:14 and 22:42 UTC**, and zero API errors in the last response. The model provenance label identified Aiden's historical fills; market and wallet models were separately connected. A per-prediction entry included a market ID, capture time, contemporary market price, model probability, input features, model source, model SHA-256 hash, and hash-chain fields.

All **144** saved snapshots came **before scheduled end**; **24** had nonzero `skill_flow`. Two crucial limitations: (1) eight markets are not 144 independent events; (2) scheduled end is not a proof that an information release, closing, or actual resolution did not happen earlier. The collector must remain running on an internet-connected computer to keep writing new entries. The ZIP cannot host a live site automatically.

## 7. What does walk-forward add?

The new `test_walk_forward_polyinsiders.py` repeats historical evaluation across calendar months. For each test month it trains only on events fully resolved before that month begins; it refits the model and predicts later market snapshots in that month. It separately evaluates the full model, model without historical skill flow, recalibrated price-only model, and raw market price. It resamples clusters of related markets at the event level when constructing uncertainty intervals. It is important to report skipped windows or unevaluated markets, not only months that generated scores.

This script is **included and syntax/test checked**, but **has not been run on the user's real `data/run/asof_features.csv` within this downloadable package**. The exact command is:

```bash
python test_walk_forward_polyinsiders.py --features data/run/asof_features.csv
```

It writes to `data/walk_forward_evaluation/`; it does **not** change the live model. Do not treat the hypothetical walk-forward result as real data until the script finishes and its `results.json` is inspected.

## 8. What should we say to the TA?

“Our goal was to test whether historically successful wallets help predict the outcomes of markets that have not resolved yet. We imported 15.58 million past fills, prepared time-stamped snapshots for 1,175 markets, trained a market-level and a wallet-level model, and connected the models to a website that records live market probabilities and our forecasts. On a held-out historical test, our market model did not improve on the prices already quoted by Polymarket. Removing the wallet-skill feature barely changed performance, and the wallet outperformance classifier was near chance. The system does record live forecasts, but evaluating those forecasts prospectively requires verified outcomes after the markets settle. We are careful not to confuse past trading skill with private information.”

The strongest part of the project is **transparent comparison with baselines and timestamped prospective logging**, not a claim of discovering actual insiders.

## 9. Remaining caveats

- Actual release and resolution timestamps need verification. Scheduled end-date checks are weaker.
- A 48-hour taker-fill sample is not a full position ledger. No historical resting order books were collected.
- Related markets within an event must remain grouped; wallet repetition across folds reduces independence.
- Market-price proxies from historical fills and current public API prices may be constructed differently (train/serve mismatch).
- A SHA-256 hash chain detects alteration to the local sequence but is not independent proof of capture time.
- Only score the earliest eligible recorded forecast per market once a verified resolution becomes available; report forecasts still pending separately.
- The team’s earlier calibrated multiple-testing wallet flag test and our current supervised wallet model use different targets and scores. Their reported AUCs must never be swapped.

## 10. How the final package should be used

If the real models are already running on your Mac, **do not overwrite `data/run/` or `web/data/live.json`** by unzipping a new full project over your working folder. Use `FINAL_UPDATE_ONLY.zip` and its safe updater. The new full ZIP contains synthetic demo models in `data/run` because the real Mac-trained model files were not sent here. The report and PowerPoint deliberately label terminal-reported metrics and the pending walk-forward test.


### Completed walk-forward results

On real historical features, 1,127 markets / 480 distinct events were evaluated over 10 months. AUC was 0.933130 with wallet-skill flow versus 0.933949 for raw market prices. Event-bootstrap confidence intervals for incremental wallet-skill contribution included zero. See `WALK_FORWARD_RESULTS_2026-11-16.json`.
