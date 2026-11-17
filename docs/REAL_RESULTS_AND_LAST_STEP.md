# PolyInsiders — actual results and last check (Nov 16, 2026)

**Results are transcribed from terminal outputs supplied by the team on Nov 16, 2026. They have not been independently recomputed in this distribution.** The zip contains the synthetic demo model by necessity; do not replace the **real-trained** `data/run/` folder already on your Mac.

## Data now prepared and verified by the user's printout

- Archive: 15,581,978 executed fills; 1,610 markets, 548,066 wallet codes.
- Converted sample: **164,129 rows**, **1,175 markets**, **491 events**, **82,586 wallets**, zero missing values.
- Snapshots: all 1,175 before recorded resolution; median lead time 54.6 hours. **This is not proof that news was not already public.** Market-level information-time proxies frequently use scheduled end times.
- The single backtest used **97 training markets** (31 events) and **303 testing markets** (148 events); the live market model was refitted on **1,175 markets** (491 events). Do not report that the test used 1,175 markets.

## Models on actual archived records

| Historical test | Model | Market price baseline |
|---|---:|---:|
| Market outcome AUC (higher better) | 0.9008 | 0.9015 |
| Market outcome Brier (lower better) | 0.1276 | 0.1244 |
| Market outcome log loss (lower better) | 0.4065 | 0.3908 |

Wallet future-outperformance model: 504 held-out wallets; **AUC 0.5109**; Brier **0.2519** versus constant-rate baseline **0.2510**. No demonstrated reliable future-wallet edge in this supervised specification.

## Wallet-skill ablation

| Specification | AUC | Brier | Log loss |
|---|---:|---:|---:|
| With wallet-skill flow | 0.9008 | 0.1276 | 0.4065 |
| Without wallet-skill flow | 0.9009 | 0.1275 | 0.4063 |
| Recalibrated price only | 0.9015 | 0.1262 | 0.4031 |
| Raw market price | 0.9015 | 0.1244 | 0.3908 |

With-minus-without wallet skill: Brier +0.000026, 95% event-bootstrap CI [-0.000057, +0.000119]; AUC -0.000092, CI [-0.000533, +0.000267]. Both CIs include zero: **no detectable incremental benefit**. Three hundred of the 303 historical test markets have nonzero wallet-skill flow, so absence of a signal is not merely due to missing historical flow.

## Live prospective tracker status

- Local machine: 144 timestamped forecasts from 8 unique markets captured Nov 16, 2026, 22:14–22:42 UTC; no API errors on the final displayed snapshot.
- All 144 entries precede their **scheduled** market end dates. A different check is still needed for actual resolution and public release timing.
- 24/144 snapshots have a nonzero wallet-skill-flow feature. The audit log contains SHA-256 linked entries; hash chains do not constitute independent trusted timestamps.
- These repeated snapshots are not 144 independent market-level validation cases. Score the *first* saved pre-resolution forecast per market after independent resolution verification.

## Completed walk-forward test on actual historical feature file

User's November 16, 2026 terminal run reports 1,127 markets, 480 distinct events, 10 monthly test windows, and 48 prepared markets not evaluated. AUC: 0.933130 with wallet-skill flow; 0.933187 without; 0.933072 recalibrated prices; 0.933949 raw Polymarket market price. Brier: 0.0972, 0.0972, 0.0972, 0.0969 respectively; log loss: 0.3046, 0.3043, 0.3039, 0.3034.

Wallet-skill flow changed Brier by +0.0000621 (95% event-bootstrap CI [-0.0001228, +0.0002422]) and AUC by -0.000057 (95% CI [-0.000345, +0.000242]). Both intervals include zero. Raw prices still slightly outperform the tested models. The test did not overwrite the live-tracker models.

Reproduction command: `python test_walk_forward_polyinsiders.py --features data/run/asof_features.csv`

The real data and outputs remain on the team's local Mac; metrics here are transcribed from supplied terminal output. This test does not independently verify information-release timing or the accuracy of prospectively logged predictions after settlement.

Other scripts included at project root: `import_aiden_parquet.py`, `test_wallet_skill_contribution.py`. The main test suite previously passed 16 tests on the user's Mac.

## Critical submission distinction

**Completed:** real historical model training, retrospective single-split evaluation, wallet-skill ablation, and a functioning local live API tracker with recorded forecasts. **Not yet established:** useful incremental prediction beyond market price, confirmed access to nonpublic information, verified feature/event-time leakage audit, or scored prospective results following settlement.
