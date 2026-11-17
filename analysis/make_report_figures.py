"""Statistics, graphs and tables for the writeup.

    python run_flagging_v2.py --data data_events/live --placebo data_placebo/live   # once
    python make_report_figures.py

Reads results_v2/ (and the two milestone1_report.json files for the data table).
Writes ONLY to results_v2/report/:
    fig1..fig7  (.png at 300 dpi for Word, .pdf for LaTeX)
    table1..table4 (.csv) and tables.md (paste-ready)
    captions.md     plain-language caption for every figure
    key_numbers.md  every number you are likely to quote, with a one-line meaning
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from pm473 import model_eval as me, skill_test as st

EV, PL, GREY, FLAG = "#3b5bdb", "#f08c00", "#98a2b3", "#e8590c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 100,
                     "savefig.bbox": "tight", "font.family": "DejaVu Sans"})


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _ci_mean(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x))


# ------------------------------------------------------------------------- figures
def fig1_data(evs: dict, out: Path, split, min_events):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    split = pd.Timestamp(split, tz="UTC")
    for name, ev, c in (("events", evs["events"], EV), ("placebo", evs["placebo"], PL)):
        e = ev.drop_duplicates("event_key")[["event_key", "period_time"]]
        mon = pd.to_datetime(e["period_time"], utc=True).dt.strftime("%Y-%m").value_counts().sort_index()
        ax[0].plot(mon.index, mon.values, marker="o", color=c, label=f"{name} ({len(e):,} events)")
    ax[0].axvline(split.strftime("%Y-%m"), color="k", ls="--", lw=1)
    ax[0].text(split.strftime("%Y-%m"), ax[0].get_ylim()[1] * 0.95, " cut-off", va="top")
    ax[0].set(title="Events resolved per month", ylabel="events (log scale)", yscale="log")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].legend(frameon=False)
    for name, ev, c in (("events", evs["events"], EV), ("placebo", evs["placebo"], PL)):
        n = ev.groupby("wallet").size()
        bins = np.unique(np.logspace(0, np.log10(max(n.max(), 2)), 30).astype(int))
        ax[1].hist(n, bins=bins, histtype="step", color=c, lw=2, label=name)
    ax[1].axvline(min_events, color="k", ls="--", lw=1)
    ax[1].text(min_events * 1.08, ax[1].get_ylim()[1] * 0.9, f"{min_events}-event minimum")
    ax[1].set(title="Events traded per wallet", xlabel="events traded (log scale)", ylabel="wallets (log scale)",
              xscale="log", yscale="log")
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig1_data_overview")


def fig2_calibration(evs, out, split, min_events):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    split = pd.Timestamp(split, tz="UTC")
    zs = {}
    for name, ev in evs.items():
        f = st.flag_wallets(st.wallet_stats(ev[ev["period_time"] < split], min_events))
        zs[name] = f
    x = np.linspace(-5, 11, 300)
    for name, c in (("events", EV), ("placebo", PL)):
        s = zs[name]
        z = s.loc[s["eligible"], "Z_adj"].to_numpy()
        ax[0].hist(z, bins=np.arange(-5, 11.5, 0.5), density=True, histtype="step", color=c, lw=2,
                   label=f"{name} ({len(z):,} wallets, {int(s['informed'].sum())} flagged)")
        qn = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        ax[1].plot(qn, np.sort(z), ".", ms=3, color=c, alpha=0.6, label=name)
    ax[0].plot(x, stats.norm.pdf(x), "k--", lw=1.3, label="luck alone: N(0, 1)")
    ax[0].set(title="Wallet scores before the cut-off", xlabel="evidence score", ylabel="density (log scale)",
              yscale="log", ylim=(1e-4, 1), xlim=(-5, 11))
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot([-4, 4], [-4, 4], "k--", lw=1)
    ax[1].set(title="Scores vs what luck alone predicts", xlabel="expected score if only luck (normal quantile)",
              ylabel="observed score", xlim=(-4, 4), ylim=(-6, 11))
    ax[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig2_score_calibration")
    return zs


def fig3_persistence(perf: dict, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for name, c in (("events", EV), ("placebo", PL)):
        p = perf[name][1]
        g = p.groupby("decile")["eval_mean_z"]
        m, e = g.mean(), g.apply(_ci_mean)
        ax[0].errorbar(m.index + (0.08 if name == "placebo" else -0.08), m.values, yerr=e.values, marker="o",
                       color=c, capsize=3, lw=1.8, label=name)
        if p["beat_prices"].nunique() > 1:
            fpr, tpr, _ = roc_curve(p["beat_prices"], p["Z"])
            ax[1].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(p['beat_prices'], p['Z']):.2f})")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(title="Later performance by score group", xlabel="score group before cut-off (10 = highest)",
              ylabel="average later result (0 = matches prices)", xticks=range(1, 11))
    ax[0].legend(frameon=False)
    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    ax[1].set(title="Can the score pick later winners?", xlabel="false positive rate", ylabel="true positive rate")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig3_persistence")


def walk_rows(evs, starts, window_days, kw):
    rows = []
    for name, ev in evs.items():
        for s0 in starts:
            end = pd.Timestamp(s0, tz="UTC") + pd.Timedelta(days=window_days)
            m, _, _ = me.performance(ev, s0, eval_end=end, **kw)
            sp = m.get("spearman_Z_vs_eval_mean_z") or {}
            rows.append({"sample": name, "flag_before": s0, "flagged": m["flagged"],
                         "rank_corr": sp.get("rho"), "ci_lo": (sp.get("ci95") or [None, None])[0],
                         "ci_hi": (sp.get("ci95") or [None, None])[1],
                         "auc": (m.get("auc_beat_prices") or {}).get("Z"),
                         "wallets_scored": m["wallets_scored_out_of_sample"]})
    return pd.DataFrame(rows)


def fig4_walk(wf: pd.DataFrame, out):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    starts = list(dict.fromkeys(wf["flag_before"]))
    x = np.arange(len(starts))
    for k, (name, c) in enumerate((("events", EV), ("placebo", PL))):
        d = wf[wf["sample"] == name].set_index("flag_before").loc[starts]
        off = (k - 0.5) * 0.36
        ax[0].bar(x + off, d["rank_corr"], 0.34, color=c, label=name,
                  yerr=[d["rank_corr"] - d["ci_lo"], d["ci_hi"] - d["rank_corr"]], capsize=3)
        ax[1].bar(x + off, d["auc"], 0.34, color=c, label=name)
    for a, t, y in ((ax[0], "Rank correlation, score vs later result", "rank correlation (0 = none)"),
                    (ax[1], "Ranking accuracy (AUC)", "AUC (0.5 = coin flip)")):
        a.set_xticks(x, [f"flag before\n{s}" for s in starts])
        a.set(title=t, ylabel=y)
    ax[1].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[1].set_ylim(0.4, max(0.75, wf["auc"].max() + 0.03))
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig4_walk_forward")


def fig5_sensitivity(rep: dict, out):
    rows = []
    for name in ("events", "placebo"):
        for r in rep.get(f"sensitivity_{name}", []):
            ci = r.get("eval_mean_z_diff_ci95") or [np.nan, np.nan]
            rows.append({"sample": name, "setting": f"split {r['split'][5:]} | min {r['min_events']} | {r['method']} q={r['q']}",
                         "flagged": r["flagged"], "diff": r["eval_mean_z_diff_vs_matched"], "lo": ci[0], "hi": ci[1]})
    d = pd.DataFrame(rows).dropna(subset=["diff"])
    if d.empty:
        return pd.DataFrame()
    order = d[d["sample"] == "events"].sort_values("diff")["setting"].tolist()
    fig, ax = plt.subplots(figsize=(8, 9))
    for name, c, off in (("events", EV, -0.18), ("placebo", PL, 0.18)):
        x = d[d["sample"] == name].set_index("setting").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(x["diff"], y, xerr=[x["diff"] - x["lo"], x["hi"] - x["diff"]], fmt="o", ms=3.5, color=c,
                    lw=1, capsize=2, label=name)
    ax.axvline(0, color="k", lw=0.9)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set(title="Flagged wallets vs matched wallets, later result, across 36 settings",
           xlabel="flagged minus matched wallets: average later event score (95% interval)")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig5_sensitivity")
    return d


def fig6_precision(perf, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for name, c in (("events", EV), ("placebo", PL)):
        m = perf[name][0]
        pk = m.get("precision_at_k_beat_prices", {})
        ks = [int(k) for k in pk]
        ax.plot(ks, [pk[str(k)] for k in ks], marker="o", color=c, lw=2, label=name)
        ax.axhline(m["base_rate_beat_prices"], color=c, ls=":", lw=1.2)
    ax.set(title="Of the top-k scores, how many beat prices later?", xlabel="k (top-scoring wallets)",
           ylabel="share that beat prices later", xscale="log", ylim=(0.4, 1.0))
    ax.text(0.02, 0.04, "dotted lines: share for all wallets", transform=ax.transAxes, fontsize=8, color="#667085")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig6_precision_at_k")


def fig7_synthetic(rep: dict, out):
    syn = rep.get("synthetic")
    if not syn:
        return pd.DataFrame()
    n = pd.DataFrame(syn["noise_only"]); p = pd.DataFrame(syn["planted"])
    t = pd.DataFrame([
        {"world": "no skilled traders", "method": m, "wallets_flagged": g["flagged"].mean(),
         "flag_rate": g["flag_rate"].mean(), "recall": np.nan, "precision": np.nan}
        for m, g in n.groupby("method")] + [
        {"world": "60 planted skilled traders", "method": m, "wallets_flagged": g["flagged"].mean(),
         "flag_rate": g["flag_rate"].mean(), "recall": g["recall"].mean(), "precision": g["precision"].astype(float).mean()}
        for m, g in p.groupby("method")])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    q = p.groupby("method")[["recall", "precision"]].agg(lambda s: pd.to_numeric(s).mean())
    x = np.arange(len(q))
    ax.bar(x - 0.18, q["recall"], 0.34, color=EV, label="share of skilled traders found")
    ax.bar(x + 0.18, q["precision"].fillna(0), 0.34, color=FLAG, label="share of flags that are skilled")
    ax.set_xticks(x, [f"{m}" for m in q.index])
    ax.set(title="Simulation check: planted skilled traders", ylabel="share", ylim=(0, 1.05))
    ax.text(0.02, 0.97, f"false flags with no skilled traders: {n[n['method'] == 'BH']['flagged'].mean():.1f} per run (BH)",
            transform=ax.transAxes, fontsize=8, va="top", color="#667085")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, out, "fig7_simulation_check")
    return t


# ------------------------------------------------------------------------- tables
def table1_data(paths: dict) -> pd.DataFrame:
    rows = []
    for name, p in paths.items():
        f = Path(p) / "milestone1_report.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        rows.append({"sample": name, "markets": r["cleaning"]["markets"], "trades": r["cleaning"]["clean_rows"],
                     "wallets": r["cleaning"]["wallets"],
                     "volume_usd_billions": r["pnl_sanity"]["sum_volume"] / 1e9,
                     "formation_markets": r.get("split", {}).get("formation_markets"),
                     "evaluation_markets": r.get("split", {}).get("evaluation_markets")})
    return pd.DataFrame(rows)


def table2_main(perf) -> pd.DataFrame:
    rows = []
    for name in ("events", "placebo"):
        m = perf[name][0]
        sp, mc = m["spearman_Z_vs_eval_mean_z"], m.get("matched_comparison", {})
        ci = mc.get("mean_z_diff_vs_matched_ci95") or [np.nan, np.nan]
        rows.append({"sample": name, "wallets_scored": m["wallets_scored_out_of_sample"], "flagged": m["flagged"],
                     "rank_corr": sp["rho"], "rank_corr_ci_lo": sp["ci95"][0], "rank_corr_ci_hi": sp["ci95"][1],
                     "auc_score": m["auc_beat_prices"]["Z"], "auc_volume_only": m["auc_beat_prices"]["volume_only"],
                     "flagged_beat_prices_later": m.get("flagged_precision_beat_prices"),
                     "all_wallets_beat_prices_later": m["base_rate_beat_prices"],
                     "flagged_minus_matched": mc.get("mean_z_diff_vs_matched"),
                     "flagged_minus_matched_ci_lo": ci[0], "flagged_minus_matched_ci_hi": ci[1],
                     "share_of_edge_that_persists": m.get("persistence_ratio"),
                     "top_minus_bottom_group": m.get("top_minus_bottom_decile_mean_z")})
    return pd.DataFrame(rows)


def table3_sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    out = []
    for name, g in d.groupby("sample"):
        out.append({"sample": name, "settings": len(g), "positive_difference": int((g["diff"] > 0).sum()),
                    "interval_above_zero": int((g["lo"] > 0).sum()),
                    "median_difference": float(g["diff"].median()),
                    "min_flagged": int(g["flagged"].min()), "max_flagged": int(g["flagged"].max())})
    return pd.DataFrame(out)


def md_table(df: pd.DataFrame, nd=3) -> str:
    if df is None or df.empty:
        return "_(not available)_\n"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    head = "| " + " | ".join(d.columns) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n"
    return head + "\n".join("| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)) + "\n"


# ------------------------------------------------------------------------- text
def write_text(out: Path, perf, wf, t1, t3, zs, args):
    e, p = perf["events"][0], perf["placebo"][0]
    es, ps = e["spearman_Z_vs_eval_mean_z"], p["spearman_Z_vs_eval_mean_z"]
    mc = e.get("matched_comparison", {})
    ci = mc.get("mean_z_diff_vs_matched_ci95") or [np.nan, np.nan]
    pmc = p.get("matched_comparison", {})
    pci = pmc.get("mean_z_diff_vs_matched_ci95") or [np.nan, np.nan]
    k = []
    k.append("# Key numbers for the writeup\n")
    k.append(f"Cut-off: wallets are scored on events that resolved before {e['split']}, then checked on events after it. "
             f"Wallets need at least {args.min_events} events to be scored and at least {args.min_eval_events} later events to be checked.\n")
    k.append("## Main result: does the score predict later results?\n")
    k.append(f"- **Rank correlation (events): {es['rho']:.2f}**, 95% CI {es['ci95'][0]:.2f} to {es['ci95'][1]:.2f}. "
             f"Placebo: **{ps['rho']:.2f}**, CI {ps['ci95'][0]:.2f} to {ps['ci95'][1]:.2f}. "
             "Meaning: wallets with higher scores before the cut-off tend to beat prices by more afterwards; 0 would mean no relationship.")
    k.append(f"- **AUC (events): {e['auc_beat_prices']['Z']:.2f}** vs {e['auc_beat_prices']['volume_only']:.2f} using trading volume alone; "
             f"placebo {p['auc_beat_prices']['Z']:.2f}. Meaning: the chance that a randomly chosen wallet that beat prices later "
             "had a higher score than one that did not. 0.5 is a coin flip.")
    if "flagged_precision_beat_prices" in e:
        k.append(f"- **Flagged wallets ({e['flagged']}) that beat prices later: {e['flagged_precision_beat_prices']:.0%}** vs "
                 f"{e['rest_rate_beat_prices']:.0%} for other wallets (Fisher p = {e['fisher_p_one_sided']:.4f}). "
                 f"Placebo: {p.get('flagged_precision_beat_prices', float('nan')):.0%} of {p['flagged']} flagged wallets.")
    k.append(f"- **Flagged minus matched wallets (average later event score): {mc.get('mean_z_diff_vs_matched', float('nan')):+.2f}**, "
             f"95% CI {ci[0]:+.2f} to {ci[1]:+.2f}. Placebo: {pmc.get('mean_z_diff_vs_matched', float('nan')):+.2f}, CI {pci[0]:+.2f} to {pci[1]:+.2f}. "
             "The flagged groups are small, so these intervals are wide; the rank correlation uses every wallet and is the stronger evidence.")
    k.append(f"- **Share of the earlier edge that persists:** {e.get('persistence_ratio', float('nan')):.2f} (1 = all of it, 0 = none).")
    k.append(f"- **Top score group minus bottom group, later result:** {e['top_minus_bottom_decile_mean_z']:+.2f} (events) vs "
             f"{p['top_minus_bottom_decile_mean_z']:+.2f} (placebo).")
    de, dp = perf["events"][2], perf["placebo"][2]
    if len(de) and len(dp):
        k.append(f"- **Where the difference comes from:** the top-scoring group's later result is {de['mean_eval_mean_z'].iloc[-1]:+.2f} "
                 f"(events) vs {dp['mean_eval_mean_z'].iloc[-1]:+.2f} (placebo), so high scorers do about equally well in both samples. "
                 f"The bottom group's later result is {de['mean_eval_mean_z'].iloc[0]:+.2f} (events) vs {dp['mean_eval_mean_z'].iloc[0]:+.2f} (placebo): "
                 "in the main sample, low scorers keep underperforming, which is what makes the overall ranking stronger there.")
    if "precision_at_k_beat_prices" in e:
        k.append("- **Top-k hit rate (events):** " + ", ".join(f"top {kk}: {v:.0%}" for kk, v in e["precision_at_k_beat_prices"].items())
                 + f"; all wallets: {e['base_rate_beat_prices']:.0%}.")
    k.append("\n## Does it hold in every period? (walk-forward)\n")
    for name in ("events", "placebo"):
        d = wf[wf["sample"] == name]
        k.append(f"- {name}: rank correlation " + ", ".join(f"{r.rank_corr:.2f} (flag before {r.flag_before})" for r in d.itertuples()))
    k.append("\n## False flags and sensitivity\n")
    for name in ("events", "placebo"):
        s = zs[name]
        k.append(f"- {name}: {int(s['informed'].sum())} of {int(s['eligible'].sum())} eligible wallets flagged "
                 f"({s['informed'].sum() / max(1, s['eligible'].sum()):.2%}). Before the fix the placebo flagged 4.5%.")
    if t3 is not None and not t3.empty:
        for r in t3.itertuples():
            k.append(f"- {r.sample}: across {r.settings} settings the flagged-minus-matched difference was positive in "
                     f"{r.positive_difference} and its interval was above zero in {r.interval_above_zero} "
                     f"(median {r.median_difference:+.2f}; {r.min_flagged} to {r.max_flagged} wallets flagged).")
    k.append("\n## How to word it\n")
    k.append("- Say \"likely informed\" or \"persistently beat market prices\", not \"insiders\".")
    k.append("- The placebo is not a perfect zero: top-scoring crypto wallets keep winning about as much as top-scoring event wallets, plausibly through speed. "
             "The test finds persistent outperformance; it cannot tell information apart from speed or other skill, so do not claim private information.")
    k.append("- The score distributions before the cut-off look alike in both samples (Figure 2); the evidence for the method is persistence after the cut-off (Figures 3 and 4), not the number of flags.")
    k.append("- Report the number of wallets behind every comparison; flagged groups are small (about 20 to 30).")
    (out / "key_numbers.md").write_text("\n".join(k) + "\n")

    c = ["# Figure captions (plain language)\n",
         "**Figure 1. The data.** Left: how many events resolved each month in the main sample (elections, economics, Fed) and in the crypto placebo; the dashed line is the date we split earlier from later events. Right: how many events each wallet traded; wallets need at least 10 to be scored.\n",
         "**Figure 2. Scores before the cut-off.** Each wallet gets an evidence score for how much its past results beat market prices. If only luck were at work, scores would follow the dashed bell curve. Both samples have a similar right tail of high scorers beyond what luck predicts, so the distribution alone does not separate them. Right: the same comparison as a quantile plot, where points above the diagonal are higher than luck predicts.\n",
         "**Figure 3. Do scores predict later results?** Wallets are ranked by score using only earlier events and split into ten groups. Left: how much each group beat prices on later events (dots are averages, bars are 95% intervals; 0 means matching prices). In the main sample the line rises across the groups; the top group does about as well in both samples, and the main sample's lower groups keep underperforming. Right: ROC curves for using the score to pick wallets that beat prices later; the diagonal is a coin flip.\n",
         "**Figure 4. Is it stable over time?** The test repeated for three periods, each time scoring wallets only on events before the date shown. Left: rank correlation between score and later results. Right: AUC.\n",
         "**Figure 5. Sensitivity.** Flagged wallets minus matched wallets (similar volume and activity) on later events, under 36 combinations of cut-off date, minimum events, flagging rule and strictness. Dots to the right of zero mean flagged wallets did better.\n",
         "**Figure 6. Precision at k.** Of the k wallets with the highest scores, the share that beat prices later. Dotted lines show the share for all wallets.\n",
         "**Figure 7. Simulation check.** In simulated markets with 60 planted skilled traders, how many the method finds and how many of its flags are right. In simulated markets with no skilled traders it flags almost no one.\n"]
    (out / "captions.md").write_text("\n".join(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--data-events", default="data_events/live")
    ap.add_argument("--data-placebo", default="data_placebo/live")
    ap.add_argument("--split-date", default="2026-06-01")
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--min-eval-events", type=int, default=3)
    ap.add_argument("--wf-starts", nargs="*", default=["2026-04-01", "2026-06-01", "2026-08-01"])
    ap.add_argument("--wf-window-days", type=int, default=61)
    a = ap.parse_args()
    res = Path(a.results)
    out = res / "report"
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((res / "validation_report.json").read_text())
    evs = {n: pd.read_parquet(res / f"events_table_{n}.parquet") for n in ("events", "placebo")}
    kw = dict(min_events=a.min_events, min_eval_events=a.min_eval_events)
    perf = {n: me.performance(evs[n], a.split_date, **kw) for n in evs}

    fig1_data(evs, out, a.split_date, a.min_events)
    zs = fig2_calibration(evs, out, a.split_date, a.min_events)
    fig3_persistence(perf, out)
    wf = walk_rows(evs, a.wf_starts, a.wf_window_days, kw)
    fig4_walk(wf, out)
    sens = fig5_sensitivity(rep, out)
    fig6_precision(perf, out)
    t4 = fig7_synthetic(rep, out)

    t1 = table1_data({"events": a.data_events, "placebo": a.data_placebo})
    t2 = table2_main(perf)
    t3 = table3_sensitivity(sens)
    for n, t in (("table1_data", t1), ("table2_main_results", t2), ("table3_sensitivity", t3),
                 ("table4_simulation", t4), ("table5_walk_forward", wf)):
        if t is not None and len(t):
            t.to_csv(out / f"{n}.csv", index=False)
    md = ["# Tables\n", "## Table 1. Data\n", md_table(t1, 2), "## Table 2. Main results\n", md_table(t2),
          "## Table 3. Sensitivity (36 settings per sample)\n", md_table(t3),
          "## Table 4. Simulation check\n", md_table(t4), "## Table 5. Walk-forward\n", md_table(wf)]
    (out / "tables.md").write_text("\n".join(md))
    write_text(out, perf, wf, t1, t3, zs, a)
    print(f"wrote figures, tables and text to {out.resolve()}")
    print(sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
