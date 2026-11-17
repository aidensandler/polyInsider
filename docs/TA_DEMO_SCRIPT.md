# TA demo — what to show (real-data status, Nov 16, 2026)

1. Open the tracker on your Mac at `http://localhost:8000/tracker.html`. Verify the current mode and model source labels. If it is stale or cannot reach the API, describe it as the last saved snapshot, not an active live forecast.
2. Show one market: contemporaneous price, model YES probability, capture time, observed taker fills. Explain that this was generated before the market's *scheduled end*, not necessarily before every relevant announcement.
3. Explain the two models: the market model predicts YES/NO; the wallet model predicts future outperformance, **not confirmed private information**. The two targets are different.
4. Open the site's **Real results** tab. Point out the raw market-price benchmark outperformed our full model (Brier 0.1244 vs 0.1276), the wallet model was near chance (AUC 0.5109), and skill ablation showed no detectable incremental benefit.
5. Open the saved audit JSONL entry with capture timestamp and model hash; explain the chain is a local integrity aid, not third-party proof of real-time capture. The user checked 144 entries before scheduled end from 8 markets.
6. Tell the TA: the monthly walk-forward script is included and ready to run; real-data output remains pending. We will only call prospective performance validated after the markets actually resolve and externally verified timestamps are checked.

**If asked why no strong insider result:** “Private information is not observable in the wallet data. We can test persistent edge and incremental forecast information; in this specification we did not find improvement beyond Polymarket's market price.”

**If asked why 1,175 markets but only 303 test markets:** 1,175 prepared markets were used for the final live-model refit; the original held-out historical test evaluated 303 markets. The walk-forward analysis will estimate performance across additional test months without relabeling the live training history as an out-of-sample test.


### Completed walk-forward results

On real historical features, 1,127 markets / 480 distinct events were evaluated over 10 months. AUC was 0.933130 with wallet-skill flow versus 0.933949 for raw market prices. Event-bootstrap confidence intervals for incremental wallet-skill contribution included zero. See `WALK_FORWARD_RESULTS_2026-11-16.json`.
