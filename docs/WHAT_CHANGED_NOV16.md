# What we added after the TA clarified the live prediction requirement

The previous package already had a **pre-resolution binary market predictor** plus an offline tracker. It **did not** yet have a supervised wallet-level forward-outperformance model. On November 16, this package was extended with:

1. `polyinsider/wallet_model.py`: a second logistic model whose dependent variable is whether a wallet has *positive mean signed market-price advantage on future resolved events*, trained on earlier windows only.
2. Historical labels and wallet features separated by **event-level** cutoffs; all outcomes of an event must be resolved to contribute to history.
3. `polyinsider/history_download.py`: public v2 historical taker-fill retriever with keyset pagination and an explicit incomplete-download report.
4. `polyinsider/prepare.py` streaming option: disk-based SQLite grouping for multi-million-row fill CSVs without loading everything into RAM.
5. `polyinsider/live.py`: live wallets include predicted future-performance probabilities from the trained wallet model, with recent 48-hour fill filters, cursor pagination, and guarded synthetic labels.
6. `predictions_audit.jsonl`: SHA-256 chained snapshots (an integrity check, not independent timestamps), plus `verify-log`.
7. Front-end: separate wallet probabilities versus YES market forecasts, visible model provenance, and stale-feed indicator.
8. Sixteen automated tests, including end-to-end API-mock capture, disjoint temporal windows, leakage falsification, chunked import parity and log-tampering detection.

**Not achieved in this environment:** reprocessing the user's original 15.6M archived fills, establishing true resolution labels for markets not in supplied data, training either model on original public trade history, or making a genuine Polymarket API capture. Those steps require the original archival files and/or an internet-enabled machine. No synthetic result should be submitted as empirical evidence.
