"""Re-flag "likely informed" wallets with the calibrated test and validate it.

Reads the outputs of run_milestone1.py (fills_clean.parquet, markets_coded.parquet,
wallet_codes.parquet). Writes ONLY to a new folder (default results_v2/), never
into data*/.  Nothing is downloaded or deleted.

    python run_flagging_v2.py --data data_events/live --placebo data_placebo/live

Checks it runs (each is a number in results_v2/validation_report.json):
  1. Placebo flag rate        - a calibrated test flags ~0% where no private info exists.
  2. Noise-only simulation    - false-positive rate with NO informed traders planted.
  3. Planted-trader recovery  - precision / recall with 60 planted informed traders.
  4. Out-of-sample persistence- flagged wallets vs matched wallets after the split date.
  5. Sensitivity grid         - split date x min events x FDR method x q.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pm473 import skill_test as st

FILL_COLS = ["wallet", "condition_id", "ts", "size", "p0", "dir0", "usdc", "fee_usdc"]


def load_events(run_dir: Path, cache: Path | None = None):
    """Per-(wallet, event) table for one run folder (cached; it is small)."""
    mk = pd.read_parquet(run_dir / "markets_coded.parquet")
    names = pd.read_parquet(run_dir / "wallet_codes.parquet").set_index("code")["wallet"]
    if cache and cache.exists():
        return pd.read_parquet(cache), mk, names
    cols = [c for c in FILL_COLS if c in pq.read_schema(run_dir / "fills_clean.parquet").names]
    fills = pq.read_table(run_dir / "fills_clean.parquet", columns=cols).to_pandas()
    wm = st.wallet_market_terms(fills, mk)
    del fills
    ev = st.event_table(wm, mk)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        ev.to_parquet(cache, index=False)
    return ev, mk, names


def synthetic_validation(n_seeds: int = 3, **rule) -> dict:
    """Noise-only false-positive rate and planted-trader recovery, over several seeds."""
    from pm473 import clean, collect, synthetic
    out = {"noise_only": [], "planted": []}
    for seed in range(n_seeds):
        for n_inf, key in ((0, "noise_only"), (60, "planted")):
            gamma, raw, truth = synthetic.generate(n_markets=320, n_noise=1200, n_informed=n_inf,
                                                   n_mm=30, seed=473 + seed)
            markets, _ = collect.filter_markets(gamma)
            fills, _ = clean.normalize(raw, markets)
            fills, mk, names, _ = clean.encode_ids(fills, markets)
            ev = st.event_table(st.wallet_market_terms(fills, mk), mk)
            cut = ev["period_time"].quantile(0.5)
            form = ev[ev["period_time"] < cut]
            for method in ("BH", "BY"):
                s = st.flag_wallets(st.wallet_stats(form, min_events=5), q=0.10, method=method, **rule)
                s.index = names[s.index.to_numpy()].astype(str)
                s["truth"] = truth.reindex(s.index)
                el, fl = s[s["eligible"]], s[s["informed"]]
                row = {"seed": seed, "method": method, "eligible": int(len(el)), "flagged": int(len(fl)),
                       "flag_rate": float(len(fl) / max(1, len(el))),
                       "false_flags_noise_or_mm": int((fl["truth"] != "informed").sum())}
                if n_inf:
                    row["precision"] = float((fl["truth"] == "informed").mean()) if len(fl) else None
                    row["recall"] = float((fl["truth"] == "informed").sum() / max(1, (el["truth"] == "informed").sum()))
                out[key].append(row)
    return out


def sensitivity(ev: pd.DataFrame, splits, min_events, methods, qs, **rule) -> list[dict]:
    rows = []
    for sp, me, meth, q in itertools.product(splits, min_events, methods, qs):
        _, r = st.run_split(ev, sp, min_events=me, q=q, method=meth, **rule)
        e, c = r["evaluation"], r["calibration"]
        rows.append({"split": sp, "min_events": me, "method": meth, "q": q,
                     "eligible": c.get("eligible_wallets"), "flagged": c.get("flagged"),
                     "flag_rate": c.get("flag_rate"),
                     "eval_mean_z_diff_vs_matched": e.get("mean_z_diff_vs_matched"),
                     "eval_mean_z_diff_ci95": e.get("mean_z_diff_vs_matched_ci95"),
                     "eval_ppd_diff_vs_matched": e.get("profit_per_dollar_diff_vs_matched"),
                     "eval_ppd_diff_ci95": e.get("profit_per_dollar_diff_vs_matched_ci95")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events/live")
    ap.add_argument("--placebo", default="data_placebo/live")
    ap.add_argument("--out", default="results_v2")
    ap.add_argument("--split-date", default="2026-06-01")
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--q", type=float, default=0.10)
    ap.add_argument("--method", choices=["BH", "BY"], default="BH")
    ap.add_argument("--skip-synthetic", action="store_true")
    ap.add_argument("--sigma-floor", type=float, default=1.0,
                    help="lowest allowed spread of the empirical null (default 1 = original rule; 0 lets it be narrower)")
    ap.add_argument("--center", action="store_true",
                    help="also centre Z on the median of eligible wallets (Efron empirical null); default centres at 0")
    ap.add_argument("--label-all", action="store_true",
                    help="LIVE labels: use every resolved event as formation data (no evaluation "
                         "split) and write likely_informed_wallets_live.csv. Re-run on a schedule "
                         "as markets resolve; the evidence for the method comes from the normal run.")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rule = dict(sigma_floor=a.sigma_floor, center=a.center)
    if a.label_all:
        ev, _, names = load_events(Path(a.data), out / "events_table_events.parquet")
        s, res = st.run_split(ev, "2100-01-01", a.min_events, a.q, a.method, **rule)
        s.index = names.reindex(s.index).to_numpy()
        s.index.name = "wallet"
        s[s["informed"]].sort_values("Z", ascending=False).to_csv(out / "likely_informed_wallets_live.csv")
        # eligible wallets only (flagged and not), so the dashboard can show near-misses
        s[s["eligible"]].sort_values("Z_adj", ascending=False).to_csv(out / "wallet_scores_live.csv")
        print(f"live labels: {int(s['informed'].sum()):,} likely-informed of "
              f"{int(s['eligible'].sum()):,} eligible wallets -> {out / 'likely_informed_wallets_live.csv'}")
        return
    report = {"args": vars(a)}

    # ---- main sample
    print("1/4 events sample")
    ev, mk, names = load_events(Path(a.data), out / "events_table_events.parquet")
    s, res = st.run_split(ev, a.split_date, a.min_events, a.q, a.method, **rule)
    report["events"] = res
    s_out = s.copy()
    s_out.index = names.reindex(s.index).to_numpy()
    s_out.index.name = "wallet"
    s_out.sort_values("Z", ascending=False).to_csv(out / "wallet_scores.csv")
    s_out[s_out["informed"]].sort_values("Z", ascending=False).to_csv(out / "likely_informed_wallets.csv")
    print(f"   flagged {int(s['informed'].sum()):,} of {int(s['eligible'].sum()):,} eligible wallets "
          f"(sigma_hat = {s.attrs['sigma']:.2f}, centre = {s.attrs['center']:+.2f})")

    # ---- placebo
    print("2/4 placebo")
    pdir = Path(a.placebo)
    if (pdir / "fills_clean.parquet").exists():
        evp, _, _ = load_events(pdir, out / "events_table_placebo.parquet")
        sp, resp = st.run_split(evp, a.split_date, a.min_events, a.q, a.method, **rule)
        report["placebo"] = resp
        print(f"   placebo flag rate {resp['calibration']['flag_rate']:.2%} "
              f"(events: {res['calibration']['flag_rate']:.2%})")
    else:
        evp = None
        print("   no placebo folder found; skipped")

    # ---- synthetic
    if not a.skip_synthetic:
        print("3/4 synthetic checks (noise-only and planted traders)")
        report["synthetic"] = synthetic_validation(**rule)

    # ---- sensitivity
    print("4/4 sensitivity grid")
    grid = dict(splits=["2026-05-01", "2026-06-01", "2026-07-01"], min_events=[5, 10, 20],
                methods=["BH", "BY"], qs=[0.05, 0.10])
    report["sensitivity_events"] = sensitivity(ev, **grid, **rule)
    if evp is not None:
        report["sensitivity_placebo"] = sensitivity(evp, **grid, **rule)

    (out / "validation_report.json").write_text(json.dumps(report, indent=2, default=_j))
    _print_summary(report)
    print(f"\nwritten to {out.resolve()}")


def _print_summary(r):
    e = r["events"]["evaluation"]
    print("\n===== summary =====")
    print(f"events:  flagged {r['events']['calibration']['flagged']:,}  "
          f"flag rate {r['events']['calibration']['flag_rate']:.2%}")
    print(f"         out-of-sample mean z, informed vs matched: "
          f"{e.get('mean_z_diff_vs_matched', float('nan')):+.3f}  CI {e.get('mean_z_diff_vs_matched_ci95')}")
    print(f"         profit per dollar,   informed vs matched: "
          f"{e.get('profit_per_dollar_diff_vs_matched', float('nan')):+.4f}  "
          f"CI {e.get('profit_per_dollar_diff_vs_matched_ci95')}")
    if "placebo" in r:
        pe = r["placebo"]["evaluation"]
        print(f"placebo: flag rate {r['placebo']['calibration']['flag_rate']:.2%}  "
              f"mean z diff {pe.get('mean_z_diff_vs_matched', float('nan')):+.3f}  "
              f"CI {pe.get('mean_z_diff_vs_matched_ci95')}")
    if "synthetic" in r:
        n = [x for x in r["synthetic"]["noise_only"] if x["method"] == "BH"]
        pl = [x for x in r["synthetic"]["planted"] if x["method"] == "BH"]
        print(f"synthetic noise-only flag rate (BH): {np.mean([x['flag_rate'] for x in n]):.2%}")
        print(f"synthetic planted recall/precision (BH): "
              f"{np.mean([x['recall'] for x in pl]):.0%} / {np.mean([x['precision'] or 0 for x in pl]):.0%}")


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, tuple): return list(o)
    return str(o)


if __name__ == "__main__":
    main()
