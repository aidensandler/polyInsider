"""Forward test of the alert models on NON-OVERLAPPING windows (and a pooled result), with a crowd-flow control.

    python run_alert_windows.py                               # defaults below
    python run_alert_windows.py --windows 2025-07-01:2025-10-01 2026-01-01:2026-04-01

For each window [start, end): train on trades made before `start` in events resolved by `start`; test on trades
made in the window in events that resolved by `end` (so no event is split across windows). Models:
  All features | price only | crowd flow (no wallet skill) | trade + agreement (no own history) | all + agreement | all + agreement + flow
The question is whether 'other GOOD wallets are on the same side' adds anything beyond price and 'money is flowing in'.
Strategies are copied with a flat $100 stake after 0 / 5 min / 1 h, as before. Results: results_v3/alerts_windows/
(backtest per window, pooled table over all windows, deciles). Needs results_v3/alerts/candidates.parquet
(python build_alert_candidates.py --data data_events_full/live --results results_v3). Memory: use the Mac (about 20 GB).

The fresh window (events resolving from 2026-10-01) is NOT touched: the default last window ends 2026-10-01.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate_alerts as ea
from pm473 import alert_backtest as bt
from pm473 import alerts as al

DEFAULT_WINDOWS = ["2025-07-01:2025-10-01", "2025-10-01:2026-01-01", "2026-01-01:2026-04-01",
                   "2026-04-01:2026-07-01", "2026-07-01:2026-10-01"]


def pooled_table(collect: dict, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for (name, dlab), parts in collect.items():
        r = pd.concat(parts, ignore_index=True)
        if r.empty:
            continue
        sm = bt.summarize(r, n_boot)
        per = r.groupby("window")[["pnl", "stake"]].sum()
        per_roi = per["pnl"] / per["stake"]
        rows.append({"strategy": name, "delay": dlab, **sm, "windows": int(len(per)),
                     "windows_positive": int((per_roi > 0).sum()), "worst_window_roi": float(per_roi.min())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events_full/live")
    ap.add_argument("--results", default="results_v3")
    ap.add_argument("--windows", nargs="*", default=DEFAULT_WINDOWS, help="start:end pairs (YYYY-MM-DD)")
    ap.add_argument("--out-name", default="alerts_windows")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args()
    res, data = Path(a.results), Path(a.data)
    out = res / a.out_name
    out.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    c = pq.read_table(res / "alerts" / "candidates.parquet", read_dictionary=["event_key"]).to_pandas()
    c["event_key"] = c["event_key"].cat.codes.astype("int32")        # integer event ids: far less memory, same grouping
    print(f"candidates: {len(c):,} trades; adding context features ...", flush=True)
    c = al.add_context_features(c)
    ev = pd.read_parquet(res / "events_table_events.parquet")
    mk = pd.read_parquet(data / "markets_coded.parquet")
    collect: dict = {}
    for w in a.windows:
        start, end = w.split(":")
        ea.run_cutoff(c, ev, mk, data, start, out, main=False, n_boot=a.n_boot, models=ea.MODELS_WINDOWS, ctx=False,
                      test_end=end, label=f"{start}_{end}", collect=collect)
    pooled = pooled_table(collect)
    pooled.to_csv(out / "pooled_table.csv", index=False)
    import json
    parts = out / "parts"
    for stem, name in (("backtest", "backtest_table.csv"), ("deciles", "model_deciles.csv")):
        fs = sorted(parts.glob(f"{stem}_*.csv"))
        pd.concat([pd.read_csv(f) if "cutoff" in pd.read_csv(f, nrows=1) else pd.read_csv(f).assign(cutoff=f.stem.split("_", 1)[1])
                   for f in fs]).to_csv(out / name, index=False)
    json.dump({f.stem.replace("info_", ""): json.load(open(f)) for f in sorted(parts.glob("info_*.json"))},
              open(out / "run_info.json", "w"), indent=1)
    show = pooled[(pooled["delay"] == "5 min")][["strategy", "alerts", "events", "roi", "roi_lo", "roi_hi", "windows_positive",
                                                  "windows", "top_profit_wallet_share"]]
    print("\nPOOLED over windows, 5 min delay:\n", show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
