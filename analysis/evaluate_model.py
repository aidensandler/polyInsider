"""Statistics on how the flagging model performs out of sample.

    python evaluate_model.py                       # uses results_v2/events_table_*.parquet
    python evaluate_model.py --split-date 2026-06-01 --min-events 10

Run run_flagging_v2.py first (it caches the per-wallet-event tables this reads).
Writes to results_v2/eval/ only:
    model_performance.json   every metric below, for events and placebo
    model_performance.png    decile plot, ROC curve, calibration plot
    decile_table_events.csv  later performance by formation-score decile
    walk_forward_events.csv  repeated tests on successive time windows (same for placebo)
Prints a short plain-language summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pm473 import model_eval as me


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--split-date", default="2026-06-01")
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--method", choices=["BH", "BY"], default="BH")
    ap.add_argument("--min-eval-events", type=int, default=3)
    ap.add_argument("--wf-starts", nargs="*", default=["2026-04-01", "2026-06-01", "2026-08-01"])
    ap.add_argument("--wf-window-days", type=int, default=61)
    a = ap.parse_args()

    res = Path(a.results)
    out = res / "eval"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(min_events=a.min_events, q=a.q, method=a.method, min_eval_events=a.min_eval_events)
    report, panels, deciles = {"args": vars(a)}, {}, {}
    for name in ("events", "placebo"):
        f = res / f"events_table_{name}.parquet"
        if not f.exists():
            print(f"{name}: {f} not found (run run_flagging_v2.py first); skipped")
            continue
        ev = pd.read_parquet(f)
        m, p, d = me.performance(ev, a.split_date, **kw)
        wf = me.walk_forward(ev, a.wf_starts, a.wf_window_days, **kw)
        report[name] = {"single_split": m, "walk_forward": wf.to_dict(orient="records")}
        panels[name], deciles[name] = p, d
        wf.to_csv(out / f"walk_forward_{name}.csv", index=False)
        if len(d):
            d.to_csv(out / f"decile_table_{name}.csv", index=False)
    if panels:
        me.make_figures(panels, deciles, out)
    (out / "model_performance.json").write_text(json.dumps(report, indent=2, default=_j))
    _summary(report)
    print(f"\nwritten to {out.resolve()}")


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _summary(r):
    for name in ("events", "placebo"):
        if name not in r:
            continue
        m = r[name]["single_split"]
        print(f"\n===== {name} (flag before {m['split']}, test after) =====")
        if "note" in m:
            print("  ", m["note"])
            continue
        print(f"  wallets flagged: {m['flagged']}  |  wallets scored out of sample: {m['wallets_scored_out_of_sample']}")
        s = m["spearman_Z_vs_eval_mean_z"]
        print(f"  rank correlation, formation score vs later performance: {_fmt(s['rho'])}  "
              f"(95% CI {_fmt(s['ci95'][0])} to {_fmt(s['ci95'][1])}; 0 = no relationship)")
        a = m["auc_beat_prices"]
        print(f"  AUC for 'beats prices later': score {_fmt(a['Z'])}  vs volume only {_fmt(a['volume_only'])}  "
              f"(0.5 = coin flip)")
        if "flagged_precision_beat_prices" in m:
            print(f"  flagged wallets that beat prices later: {m['flagged_precision_beat_prices']:.0%} "
                  f"vs {m['rest_rate_beat_prices']:.0%} for the rest  (lift {_fmt(m['lift'], 2)}, "
                  f"Fisher p = {_fmt(m['fisher_p_one_sided'], 4)})")
            print(f"  mean later event z: flagged {_fmt(m['flagged_eval_mean_z'])} vs rest {_fmt(m['rest_eval_mean_z'])}; "
                  f"share of the formation edge that persists: {_fmt(m['persistence_ratio'], 2)}")
        mc = m.get("matched_comparison", {})
        if "mean_z_diff_vs_matched" in mc:
            ci = mc["mean_z_diff_vs_matched_ci95"]
            print(f"  flagged minus matched wallets (mean z): {_fmt(mc['mean_z_diff_vs_matched'])} "
                  f"(95% CI {_fmt(ci[0])} to {_fmt(ci[1])})")
        print(f"  top decile minus bottom decile (later mean z): {_fmt(m.get('top_minus_bottom_decile_mean_z'))}")
        print("  precision at k:", {k: round(v, 2) for k, v in m["precision_at_k_beat_prices"].items()})
        wf = pd.DataFrame(r[name]["walk_forward"])
        if len(wf):
            print("  walk-forward (flag on the past, test on the next window):")
            print(wf[["split", "flagged", "scored_flagged", "auc_Z", "spearman", "diff_vs_matched"]]
                  .round(3).to_string(index=False))


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
