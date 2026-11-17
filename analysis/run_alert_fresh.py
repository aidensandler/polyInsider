"""ONE-SHOT confirmatory test of the frozen alert rule on the fresh window (trades from 2026-10-01 on).

    python run_alert_fresh.py

Frozen before this run (set in an earlier version of evaluate_alerts.py, not tuned on any window):
  PRIMARY rule: copy a trade by a wallet with a past score Z >= 3 over at least 10 resolved events
  (history strictly before the trade), one alert per (wallet, market, outcome), $100 flat, price of the
  next trade 5 minutes later, held to resolution. Secondary (reported, not primary): trading with wallets
  flagged on events resolved before the window, under the original rule and the less conservative rule.
Compared with random candidate trades under the same copying rules. Nothing is trained, so nothing can leak.
Writes results_v3/alerts_fresh/. Reads results_v3/alerts/candidates.parquet (filtered to the window).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pm473 import alert_backtest as bt
from pm473 import skill_test as st

DELAYS = {"0 min (upper bound)": 0, "5 min": 300, "1 hour": 3600}
COLS = ["wallet", "condition_id", "event_key", "ts", "event_time", "size", "usdc", "payout", "edge", "entry_price",
        "held_outcome", "n_past", "Z_past"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events_full/live")
    ap.add_argument("--results", default="results_v3")
    ap.add_argument("--start", default="2026-10-01")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    res, data = Path(a.results), Path(a.data)
    out = res / "alerts_fresh"; out.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start, tz="UTC")
    te = pq.read_table(res / "alerts" / "candidates.parquet", columns=COLS,
                       filters=[("ts", ">=", start)]).to_pandas()
    te["event_key"] = te["event_key"].astype(str)
    print(f"fresh window: {len(te):,} candidate trades from {te['ts'].min()} to {te['ts'].max()}; "
          f"{te['event_key'].nunique()} events", flush=True)
    ev = pd.read_parquet(res / "events_table_events.parquet")
    form = ev[ev["period_time"] < start]
    tape = bt.make_tape(data / "fills_clean.parquet", te["condition_id"].unique(), start)

    strategies = {"PRIMARY: past score Z >= 3 (10+ events)": bt.dedupe_alerts(te[(te["Z_past"] >= 3) & (te["n_past"] >= 10)])}
    info = {"start": a.start, "trades": int(len(te)), "events": int(te["event_key"].nunique())}
    for name, kw in (("Secondary: flagged wallets, original rule", dict(sigma_floor=1.0, center=False)),
                     ("Secondary: flagged wallets, less conservative rule", dict(sigma_floor=0.0, center=True))):
        s = st.flag_wallets(st.wallet_stats(form, 10), q=0.10, method="BH", **kw)
        codes = set(s.index[s["informed"]])
        strategies[name] = bt.dedupe_alerts(te[te["wallet"].isin(codes)])
        info[name] = {"flagged_wallets": len(codes)}
    pool = bt.dedupe_alerts(te.sample(min(len(te), 600_000), random_state=0))
    rows = []
    for slip in (0.0, 0.01):
        for dlab, d in DELAYS.items():
            if slip and d == 0:
                continue
            for name, al_ in strategies.items():
                r = bt.follower_results(al_, tape, d, 100.0, slip)
                sm = bt.summarize(r, a.n_boot)
                rb = bt.random_benchmark(pool, tape, max(1, min(len(al_), len(pool) // 2)), d, slip, 300) if len(al_) else {}
                rows.append({"strategy": name, "delay": dlab, "delay_s": d, "slippage": slip, **sm,
                             "random_mean": rb.get("mean"), "random_p05": rb.get("p05"), "random_p95": rb.get("p95")})
                if name.startswith("PRIMARY") and d == 300 and slip == 0:
                    r.assign(market=r["condition_id"]).to_parquet(out / "primary_alerts_5min.parquet", index=False)
            rr = bt.summarize(bt.follower_results(pool, tape, d, 100.0, slip), a.n_boot)
            rows.append({"strategy": "All candidate trades", "delay": dlab, "delay_s": d, "slippage": slip, **rr})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "fresh_backtest.csv", index=False)
    json.dump(info, open(out / "fresh_info.json", "w"), indent=1, default=float)
    show = tab[(tab.slippage == 0)][["strategy", "delay", "alerts", "events", "wallets", "roi", "roi_lo", "roi_hi", "win_rate",
                                      "top_profit_wallet_share", "random_mean"]]
    print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
