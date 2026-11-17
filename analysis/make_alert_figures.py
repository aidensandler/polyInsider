"""Figures and captions for the alert model, from results_v2/alerts/ (run evaluate_alerts.py first).

    python make_alert_figures.py          # -> results_v2/alerts/figures/*.png, *.pdf and captions_alerts.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, TEAL, AMBER, GREY, RED = "#3b5bdb", "#0ca678", "#e8590c", "#98a2b3", "#c92a2a"
COLORS = {"All candidate trades": GREY, "Flagged wallets' trades": TEAL, "Past score Z >= 3 alerts": BLUE, "Model alerts": AMBER}
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "font.size": 10})
TICKS_H = [1, 6, 24, 168, 720, 2160]
TICK_LABELS = ["1 h", "6 h", "1 day", "1 week", "30 days", "90 days"]


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def weighted_quantile(x, w, q):
    o = np.argsort(x); x, w = np.asarray(x)[o], np.asarray(w)[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)])


def lead_time_stats(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, x in d.groupby("group"):
        h = x["hours_to_resolve"].dropna()
        rows.append({"group": g, "trades": len(x), "median hours before resolution": h.median(),
                     "10th percentile (hours)": h.quantile(0.1), "90th percentile (hours)": h.quantile(0.9),
                     "share within 1 day": (h < 24).mean(), "share within 7 days": (h < 168).mean(), "share more than 30 days out": (h > 720).mean(),
                     "dollar-weighted median (hours)": weighted_quantile(x["hours_to_resolve"].fillna(h.median()), x["usdc"], 0.5)})
        by_w = x.groupby("wallet")["usdc"].sum().sort_values(ascending=False)
        rest = x[x["wallet"] != by_w.index[0]] if len(by_w) > 1 else x
        rows[-1]["largest wallet's share of dollars"] = float(by_w.iloc[0] / by_w.sum())
        rows[-1]["dollar-weighted median without the largest wallet (hours)"] = weighted_quantile(rest["hours_to_resolve"].fillna(h.median()), rest["usdc"], 0.5)
    return pd.DataFrame(rows).set_index("group")


def fig_lead_time(d: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, np.log10(7000), 45)
    for g in ["All candidate trades", "Flagged wallets' trades", "Past score Z >= 3 alerts", "Model alerts"]:
        x = d.loc[d["group"] == g]
        h = np.log10(x["hours_to_resolve"].clip(lower=1).dropna())
        ax[0].hist(h, bins=bins, density=True, histtype="step", lw=2, color=COLORS[g], label=f"{g} (n={len(x):,})")
        o = np.argsort(x["hours_to_resolve"].fillna(1e9).to_numpy()); hh = x["hours_to_resolve"].fillna(1e9).to_numpy()[o]
        c = np.cumsum(x["usdc"].to_numpy()[o]) / x["usdc"].sum()
        ax[1].plot(np.log10(np.clip(hh, 1, 7000)), c, lw=2, color=COLORS[g], label=g)
    for a in ax:
        a.set_xticks(np.log10(TICKS_H), TICK_LABELS)
        a.set_xlabel("time before the market resolved (log scale)")
    ax[0].set(title="When trades are made (share of trades)", ylabel="density"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].set(title="When the money goes in (share of dollars)", ylabel="share of dollars traded by this time before resolution", ylim=(0, 1))
    ax[1].invert_xaxis(); ax[1].set_ylabel("share of dollars placed this far or closer to resolution"); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
    plt.tight_layout(); save(fig, out, "alert_fig1_lead_time")


STRATS = ["All features", "Trade features only", "Wallet history only", "Price only", "Trade with flagged wallets", "Past score Z >= 3 (10+ events)"]


def fig_backtest(b: pd.DataFrame, out: Path, cutoff="2026-06-01"):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
    for a, (dl, ds) in zip(ax, (("0 min (upper bound)", "0 minutes (upper bound)"), ("5 min", "5 minutes"), ("1 hour", "1 hour"))):
        d = b[(b["cutoff"] == cutoff) & (b["delay"] == dl) & (b["slippage"] == 0)].set_index("strategy").reindex(STRATS)
        y = np.arange(len(d))[::-1]
        for yi, (s_, r) in zip(y, d.iterrows()):
            if pd.isna(r.get("roi")):
                continue
            if pd.notna(r.get("random_p05")):
                a.fill_betweenx([yi - 0.38, yi + 0.38], r["random_p05"] * 100, r["random_p95"] * 100, color=GREY, alpha=0.4, lw=0)
            a.errorbar(r["roi"] * 100, yi, xerr=[[(r["roi"] - r["roi_lo"]) * 100], [(r["roi_hi"] - r["roi"]) * 100]],
                       fmt="o", color=AMBER if s_ == "All features" else BLUE, capsize=3)
        a.axvline(0, color="k", lw=0.8)
        a.set_yticks(y, [f"{s_}\n({int(d.loc[s_, 'alerts']) if pd.notna(d.loc[s_, 'alerts']) else 0:,} alerts)" for s_ in STRATS], fontsize=8)
        a.set_title(f"Copy delay: {ds}", fontsize=10); a.set_xlabel("return per dollar staked (%)"); a.set_xlim(-25, 45)
    fig.text(0.01, -0.02, "Dot and line: return and 95% interval (resampling events). Grey band: 5th to 95th percentile of the same number of randomly chosen trades.", fontsize=8)
    plt.tight_layout(); save(fig, out, "alert_fig2_backtest")


def fig_robust(b: pd.DataFrame, out: Path):
    d = b[(b["delay_s"] == 300) & (b["slippage"] == 0) & b["strategy"].isin(STRATS)]
    cuts = sorted(d["cutoff"].unique())
    fig, ax = plt.subplots(figsize=(8, 3.8))
    w = 0.8 / max(1, len(cuts))
    for i, c in enumerate(cuts):
        x = d[d["cutoff"] == c].set_index("strategy").reindex(STRATS)
        ax.bar(np.arange(len(STRATS)) + i * w - 0.4 + w / 2, x["roi"] * 100, w, label=f"model trained on trades before {c}", color=[BLUE, TEAL, AMBER][i % 3], alpha=0.85)
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(range(len(STRATS)), [s.replace(" (10+ events)", "").replace("Trade with flagged wallets", "Flagged wallets") for s in STRATS], rotation=20, ha="right", fontsize=8)
    ax.set(title="Same test with three different training cut-offs (5 minute copy delay)", ylabel="return per dollar staked (%)"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); save(fig, out, "alert_fig3_cutoffs")


def fig_cumulative(cur: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for s_, c in (("All features", AMBER), ("Price only", GREY), ("Trade with flagged wallets", TEAL)):
        x = cur[cur["strategy"] == s_].copy(); x["day"] = pd.to_datetime(x["day"])
        ax.plot(x["day"], x["cum_roi"] * 100, lw=2, color=c, label=f"{s_} (${x['cum_stake'].iloc[-1] / 1000:,.0f}k staked in total)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="Running return per dollar staked, copying with a flat $100 per alert (5 minute delay)", ylabel="return per dollar staked so far (%)", xlabel="day the market resolved")
    ax.legend(frameon=False, fontsize=8); fig.autofmt_xdate(); plt.tight_layout(); save(fig, out, "alert_fig4_cumulative")


def fig_deciles(dec: pd.DataFrame, out: Path, cutoff="2026-06-01"):
    d = dec[dec["cutoff"] == cutoff] if "cutoff" in dec else dec
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for m, c in (("All features", AMBER), ("Price only", GREY), ("Wallet history only", TEAL)):
        x = d[d["model"] == m]
        ax[0].plot(x["decile"], x["mean_realized_edge"] * 100, "o-", color=c, label=m)
        ax[1].plot(x["decile"], x["roi_dollar_weighted"] * 100, "o-", color=c, label=m)
    for a, t, yl in ((ax[0], "Realised edge by predicted-edge decile", "average edge per share (cents)"), (ax[1], "Return per dollar by predicted-edge decile", "return per dollar traded (%)")):
        a.axhline(0, color="k", lw=0.8); a.set(title=t, xlabel="score decile on the later period (10 = highest predicted)", ylabel=yl)
    ax[0].legend(frameon=False, fontsize=8); plt.tight_layout(); save(fig, out, "alert_fig5_deciles")


CAPTIONS = {
    "alert_fig1_lead_time": "Figure A1. How long before the market resolved each group trades (after the cut-off). Left: share of trades. Right: share of dollars. 'All candidate trades' are trades of at least $5 by wallets with 3 or more resolved events. Time is measured to the actual resolution, so it can only be used to describe trades, not to score them live. Caution: one wallet places about 89% of the flagged wallets' dollars, mostly in a single event, so the right-hand curve for that group describes that wallet, not a typical flagged wallet; the table shows the figures without it.",
    "alert_fig2_backtest": "Figure A2. Return per dollar of copying each alert with a flat stake, tested only on trades after June 1, 2026. Grey bands show what the same number of randomly chosen trades would earn. Intervals resample whole events, because markets in an event are not independent.",
    "alert_fig3_cutoffs": "Figure A3. The same test repeated with models trained on trades before May 1, June 1 and July 1, 2026, and tested after. A result that appears at one cut-off only is not reliable.",
    "alert_fig4_cumulative": "Figure A4. Running return per dollar staked when copying with $100 per alert, by the day the market resolved (profit is only realised then). Early in the period few markets have resolved, so the lines are noisy; single large events can move them sharply.",
    "alert_fig5_deciles": "Figure A5. Trades after the cut-off sorted by each model's predicted edge. If a model works, realised edge and return should rise from decile 1 to 10. A price-only model is shown because a lot of apparent skill is just the price a trade was made at.",
}


def main(results="results_v2"):
    out = Path(results) / "alerts"; fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True, parents=True)
    d = pd.read_parquet(out / "lead_time_sample.parquet")
    b = pd.read_csv(out / "backtest_table.csv")
    fig_lead_time(d, fig_dir); lead_time_stats(d).to_csv(out / "lead_time_stats.csv")
    fig_backtest(b, fig_dir); fig_robust(b, fig_dir)
    fig_cumulative(pd.read_csv(out / "cumulative_pnl.csv"), fig_dir)
    dec = pd.read_csv(out / "model_deciles.csv")
    fig_deciles(dec, fig_dir)
    json.dump(CAPTIONS, open(out / "captions_alerts.json", "w"), indent=1)
    print("wrote", sorted(p.name for p in fig_dir.glob("*.png")))


if __name__ == "__main__":
    main()
