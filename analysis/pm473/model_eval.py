"""Statistics on how well the flagging model performs out of sample.

Question answered: do wallets that look skilled BEFORE a cut-off date still beat
market prices AFTER it?

Setup (one split, or several walk-forward splits)
  formation: events resolving before the split  -> compute each wallet's score Z, flag wallets
  evaluation: events resolving in the window after it -> measure what the wallet earned

Outcome per wallet in evaluation (needs >= min_eval_events events, else too noisy):
  eval_mean_z   average event z-score; 0 if the wallet only matches prices
  beat_prices   eval_mean_z > 0           (primary target)
  profitable    evaluation profit > 0 dollars (secondary target)

Everything is computed from the per-(wallet, event) table made by skill_test.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from . import skill_test as st


def _eval_table(evl: pd.DataFrame) -> pd.DataFrame:
    e = evl.dropna(subset=["z_clipped"])
    g = e.groupby("wallet")
    t = pd.DataFrame({"eval_events": g.size(), "eval_mean_z": g["z_clipped"].mean(),
                      "eval_profit": g["c"].sum(), "eval_volume": g["volume"].sum()})
    t["eval_ppd"] = t["eval_profit"] / t["eval_volume"].replace(0, np.nan)
    return t


def _auc(y, score):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, np.asarray(score, float)))


def _boot_stat(fn, df, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        vals.append(fn(df.iloc[rng.integers(0, len(df), len(df))]))
    vals = np.array([v for v in vals if np.isfinite(v)])
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) else (None, None)


def performance(ev: pd.DataFrame, split, eval_end=None, min_events: int = 10, q: float = 0.10,
                method: str = "BH", min_eval_events: int = 3, seed: int = 0):
    """Performance of one formation/evaluation split.
    Returns (metrics dict, panel DataFrame, decile table)."""
    tz = lambda x: pd.Timestamp(x) if pd.Timestamp(x).tzinfo else pd.Timestamp(x, tz="UTC")
    split = tz(split)
    form = ev[ev["period_time"] < split]
    win = ev[ev["period_time"] >= split]
    if eval_end is not None:
        win = win[win["period_time"] < tz(eval_end)]
    s = st.flag_wallets(st.wallet_stats(form, min_events), q=q, method=method)
    t = _eval_table(win)
    p = s[s["eligible"]].join(t, how="inner")
    p = p[p["eval_events"] >= min_eval_events].copy()
    m = {"split": str(split.date()), "eval_end": None if eval_end is None else str(tz(eval_end).date()),
         "eligible_wallets": int(s["eligible"].sum()), "flagged": int(s["informed"].sum()),
         "wallets_scored_out_of_sample": int(len(p)), "min_eval_events": min_eval_events}
    if len(p) < 30:
        m["note"] = "too few wallets with enough evaluation events"
        return m, p, pd.DataFrame()
    p["beat_prices"] = (p["eval_mean_z"] > 0).astype(int)
    p["profitable"] = (p["eval_profit"] > 0).astype(int)
    p["log_volume"] = np.log1p(p["volume"])
    m["base_rate_beat_prices"] = float(p["beat_prices"].mean())
    m["base_rate_profitable"] = float(p["profitable"].mean())

    # --- ranking quality: does formation Z order wallets by later performance?
    rho = stats.spearmanr(p["Z"], p["eval_mean_z"])
    lo, hi = _boot_stat(lambda d: stats.spearmanr(d["Z"], d["eval_mean_z"])[0], p, seed=seed)
    m["spearman_Z_vs_eval_mean_z"] = {"rho": float(rho[0]), "p": float(rho[1]), "ci95": [lo, hi]}

    # --- classification quality (AUC) against simple baselines
    m["auc_beat_prices"] = {"Z": _auc(p["beat_prices"], p["Z"]),
                            "volume_only": _auc(p["beat_prices"], p["log_volume"]),
                            "n_events_only": _auc(p["beat_prices"], p["n_events"]),
                            "past_profit_per_dollar": _auc(p["beat_prices"], p["profit_per_dollar"].fillna(0))}
    m["auc_profitable"] = {"Z": _auc(p["profitable"], p["Z"]),
                           "volume_only": _auc(p["profitable"], p["log_volume"])}

    # --- flagged vs everyone else
    f, r = p[p["informed"]], p[~p["informed"]]
    m["n_flagged_scored"] = int(len(f))
    if len(f) >= 5:
        a, b = int(f["beat_prices"].sum()), int(len(f) - f["beat_prices"].sum())
        c, d = int(r["beat_prices"].sum()), int(len(r) - r["beat_prices"].sum())
        odds, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
        m["flagged_precision_beat_prices"] = float(a / len(f))
        m["rest_rate_beat_prices"] = float(c / max(1, len(r)))
        m["lift"] = float((a / len(f)) / max(1e-9, p["beat_prices"].mean()))
        m["fisher_p_one_sided"] = float(pval)
        m["flagged_eval_mean_z"] = float(f["eval_mean_z"].mean())
        m["rest_eval_mean_z"] = float(r["eval_mean_z"].mean())
        # persistence: how much of the formation edge survives (1 = all, 0 = none)
        m["persistence_ratio"] = float(f["eval_mean_z"].mean() / f["mean_z"].mean()) if f["mean_z"].mean() else None
        # one-sample test: do flagged wallets beat fair prices out of sample? (null mean = 0)
        tt = stats.ttest_1samp(f["eval_mean_z"], 0.0, alternative="greater")
        m["flagged_eval_mean_z_vs_zero_p"] = float(tt.pvalue)
        m["pooled_profit_per_dollar_flagged"] = float(f["eval_profit"].sum() / f["eval_volume"].sum())
        m["pooled_profit_per_dollar_rest"] = float(r["eval_profit"].sum() / r["eval_volume"].sum())
    m["matched_comparison"] = st.evaluate(s, win)

    # --- precision at k (top-k wallets by formation Z)
    top = p.sort_values("Z", ascending=False)
    m["precision_at_k_beat_prices"] = {str(k): float(top.head(k)["beat_prices"].mean())
                                       for k in (25, 50, 100, 200, 500) if len(top) >= k}

    # --- decile table: is the relation monotone?
    p["decile"] = pd.qcut(p["Z"].rank(method="first"), 10, labels=False) + 1
    dec = p.groupby("decile").agg(wallets=("Z", "size"), mean_formation_Z=("Z", "mean"),
                                  mean_eval_mean_z=("eval_mean_z", "mean"),
                                  share_beat_prices=("beat_prices", "mean"),
                                  share_profitable=("profitable", "mean"),
                                  eval_profit=("eval_profit", "sum"), eval_volume=("eval_volume", "sum"))
    dec["pooled_profit_per_dollar"] = dec["eval_profit"] / dec["eval_volume"].replace(0, np.nan)
    m["top_minus_bottom_decile_mean_z"] = float(dec["mean_eval_mean_z"].iloc[-1] - dec["mean_eval_mean_z"].iloc[0])
    return m, p, dec.reset_index()


def walk_forward(ev: pd.DataFrame, starts: list[str], window_days: int = 61, **kw) -> pd.DataFrame:
    """Repeat the evaluation over successive non-overlapping windows:
    flag on everything before `start`, evaluate in [start, start + window_days)."""
    rows = []
    for s0 in starts:
        end = pd.Timestamp(s0, tz="UTC") + pd.Timedelta(days=window_days)
        m, _, _ = performance(ev, s0, eval_end=end, **kw)
        mc = m.get("matched_comparison", {})
        rows.append({"split": m["split"], "eval_end": m["eval_end"], "eligible": m["eligible_wallets"],
                     "flagged": m["flagged"], "scored_flagged": m.get("n_flagged_scored"),
                     "auc_Z": (m.get("auc_beat_prices") or {}).get("Z"),
                     "spearman": (m.get("spearman_Z_vs_eval_mean_z") or {}).get("rho"),
                     "flagged_eval_mean_z": m.get("flagged_eval_mean_z"),
                     "rest_eval_mean_z": m.get("rest_eval_mean_z"),
                     "diff_vs_matched": mc.get("mean_z_diff_vs_matched"),
                     "diff_ci_lo": (mc.get("mean_z_diff_vs_matched_ci95") or [None, None])[0],
                     "diff_ci_hi": (mc.get("mean_z_diff_vs_matched_ci95") or [None, None])[1]})
    return pd.DataFrame(rows)


def make_figures(panels: dict, deciles: dict, out_dir):
    """panels/deciles: {"events": ..., "placebo": ...}. Saves PNGs under out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, d in deciles.items():
        if len(d):
            ax[0].plot(d["decile"], d["mean_eval_mean_z"], marker="o", label=name)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set(title="Later performance by formation-score decile", xlabel="decile of formation Z (10 = highest)",
              ylabel="mean evaluation event z (0 = matches prices)")
    ax[0].legend()
    for name, p in panels.items():
        if len(p) and p["beat_prices"].nunique() > 1:
            fpr, tpr, _ = roc_curve(p["beat_prices"], p["Z"])
            ax[1].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(p['beat_prices'], p['Z']):.2f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    ax[1].set(title="ROC: formation Z predicting 'beats prices later'", xlabel="false positive rate",
              ylabel="true positive rate")
    ax[1].legend()
    for name, p in panels.items():
        if len(p):
            z = np.sort(p["Z"].to_numpy())
            qn = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
            ax[2].plot(qn, z, ".", ms=3, label=name, alpha=0.6)
    lim = [-4, 4]
    ax[2].plot(lim, lim, "k--", lw=0.6)
    ax[2].set(title="Calibration: formation Z vs normal (placebo should hug the line)",
              xlabel="normal quantile", ylabel="observed Z", xlim=lim, ylim=(-6, 8))
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_performance.png", dpi=130)
    plt.close(fig)
