"""Tables and figures for timing_analysis.py (reads results_v4/timing/timing_trades.parquet)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BLUE, TEAL, AMBER, GREY = "#3b5bdb", "#0ca678", "#e8590c", "#98a2b3"
LIFE_LABELS = [f"{i * 10}-{i * 10 + 10}%" for i in range(10)]
HRES_EDGES = np.array([0, 1, 6, 24, 72, 168, 720, 2160, 1e9])
HRES_LABELS = ["under 1 h", "1-6 h", "6-24 h", "1-3 days", "3-7 days", "7-30 days", "30-90 days", "90+ days"]
REL_LABELS = ["7+ days before", "1-7 days before", "7-24 h before", "2-6 h before", "the hour before", "the hour of the shift",
              "1-6 h after", "7-24 h after", "24+ h after"]
FOMC_EDGES = np.array([-1e9, -1 / 60, 0, 15 / 60, 1, 6, 24, 1e9])            # hours_to_info (positive = before the statement)
FOMC_LABELS = ["after, 1+ h", "after, 0-1 h", "0-15 min before", "15-60 min before", "1-6 h before", "6-24 h before", "24+ h before"]
FINE_LABELS = ["6+ h before", "1-6 h before", "30-60 min before", "10-30 min before", "the 10 minutes before", "the 10 minutes of the jump",
               "10-30 min after", "30-60 min after", "1-6 h after", "6+ h after"]
MIN_USD = 1000.0


def wallet_matrix(w, b, usd, nb):
    ok = b >= 0
    uw, inv = np.unique(w[ok], return_inverse=True)
    m = np.bincount(inv * nb + b[ok], weights=usd[ok], minlength=len(uw) * nb).reshape(len(uw), nb)
    return uw, m


def shares(m, min_usd=MIN_USD):
    """pooled dollar share, and wallet-equal share (each wallet counts once; wallets under min_usd left out)."""
    pooled = m.sum(0) / max(m.sum(), 1e-9)
    tot = m.sum(1)
    keep = tot >= min_usd
    eq = (m[keep] / tot[keep, None]).mean(0) if keep.any() else np.full(m.shape[1], np.nan)
    return pooled, eq, int(keep.sum())


def boot_equal(m, n=1000, seed=0, min_usd=MIN_USD):
    tot = m.sum(1); keep = tot >= min_usd
    r = m[keep] / tot[keep, None]
    if len(r) < 3:
        return np.full((2, m.shape[1]), np.nan)
    rng = np.random.default_rng(seed)
    bs = np.array([r[rng.integers(0, len(r), len(r))].mean(0) for _ in range(n)])
    return np.percentile(bs, [2.5, 97.5], axis=0)


def report(tdir: Path):
    tdir = Path(tdir); fig_dir = tdir / "figures"; fig_dir.mkdir(exist_ok=True)
    info = json.load(open(tdir / "timing_info.json"))
    t = pd.read_parquet(tdir / "timing_trades.parquet")
    f2 = (t["grp"] & 2) > 0
    other = ((t["grp"] & 1) > 0) & ~f2
    groups = {"Flagged wallets (less conservative rule)": f2, "Other experienced wallets": other}
    w, usd = t["wallet"].to_numpy(), t["usdc"].to_numpy("float64")
    life_b = np.where(t["frac_life"].notna(), np.clip((t["frac_life"].fillna(0) * 10).astype(int), 0, 9), -1)
    hres_b = np.where(t["hours_to_resolve"].notna(), np.clip(np.digitize(t["hours_to_resolve"].fillna(0), HRES_EDGES) - 1, 0, 7), -1)
    rel_b = t["rel_bucket"].to_numpy()
    clocks = {"Share of the market's life elapsed": (life_b, LIFE_LABELS, "life"),
              "Time before the market resolved": (hres_b, HRES_LABELS, "hres"),
              "Time relative to the market's biggest price shift": (rel_b, REL_LABELS, "rel"),
              "Minutes around the sharpest 10-minute jump": (t["rel10_bucket"].to_numpy(), FINE_LABELS, "rel10")}
    rows, mats, cis = [], {}, {}
    for cname, (b, labels, key) in clocks.items():
        for gname, mask in groups.items():
            uw, m = wallet_matrix(w[mask.to_numpy()], b[mask.to_numpy()], usd[mask.to_numpy()], len(labels))
            mats[(key, gname)] = m
            pooled, eq, nw = shares(m)
            if gname.startswith("Flagged"):
                cis[key] = boot_equal(m)
            ci = cis.get(key) if gname.startswith("Flagged") else None
            for i, lab in enumerate(labels):
                rows.append({"clock": cname, "group": gname, "bucket": lab, "pooled_dollar_share": pooled[i], "wallet_equal_share": eq[i],
                             "wallet_equal_lo": ci[0][i] if ci is not None else np.nan, "wallet_equal_hi": ci[1][i] if ci is not None else np.nan,
                             "wallets": nw, "dollars": float(m.sum())})
        a = np.array(info["all_traders_dollars"][key]); a = a / a.sum()
        for i, lab in enumerate(labels):
            rows.append({"clock": cname, "group": "All traders", "bucket": lab, "pooled_dollar_share": a[i], "wallet_equal_share": np.nan,
                         "wallet_equal_lo": np.nan, "wallet_equal_hi": np.nan, "wallets": np.nan, "dollars": np.nan})
    shares_df = pd.DataFrame(rows); shares_df.to_csv(tdir / "timing_clock_shares.csv", index=False)

    # markouts: price gain per dollar after the trade (dollar-weighted), and at resolution
    ok = t["entry_price"].between(0.02, 0.98)
    ret = {"5 min": t["m5m"] / t["entry_price"], "1 hour": t["m1h"] / t["entry_price"], "6 hours": t["m6h"] / t["entry_price"],
           "24 hours": t["m24h"] / t["entry_price"], "at resolution": t["edge"] / t["entry_price"]}
    mo_rows = []
    rng = np.random.default_rng(0)
    for gname, mask in groups.items():
        sel = ok & mask
        d = pd.DataFrame({"w": w[sel.to_numpy()], "u": usd[sel.to_numpy()]});
        for h, r in ret.items():
            d["x"] = (r[sel].to_numpy() * usd[sel.to_numpy()])
            g = d.groupby("w")[["x", "u"]].sum()
            val = g["x"].sum() / g["u"].sum()
            lo = hi = np.nan
            if gname.startswith("Flagged"):
                gx, gu = g["x"].to_numpy(), g["u"].to_numpy()
                bs = [gx[i].sum() / gu[i].sum() for i in (rng.integers(0, len(g), len(g)) for _ in range(1000))]
                lo, hi = np.percentile(bs, [2.5, 97.5])
            mo_rows.append({"group": gname, "horizon": h, "gain_per_dollar": val, "lo": lo, "hi": hi, "wallets": len(g)})
    mo = pd.DataFrame(mo_rows); mo.to_csv(tdir / "timing_markouts.csv", index=False)

    # where is the profit earned, and how often on the side the price then moved to (with wallet-cluster bootstrap intervals)
    def bucket_table(bcol, wcol, labels, tag):
        out = []
        for gname, mask in groups.items():
            s_ = t[mask & ok & (t[bcol] >= 0)]
            x_all = (s_["edge"] / s_["entry_price"] * s_["usdc"]).to_numpy(); u_all = s_["usdc"].to_numpy("float64")
            side = np.where(s_[wcol].to_numpy(), u_all, 0.0)
            for i, lab in enumerate(labels):
                sel = (s_[bcol] == i).to_numpy()
                if not sel.any():
                    continue
                df_ = pd.DataFrame({"w": s_["wallet"].to_numpy()[sel], "x": x_all[sel], "u": u_all[sel], "side": side[sel]}).groupby("w").sum()
                row = {"clock": tag, "group": gname, "bucket": lab, "trades": int(sel.sum()), "wallets": len(df_), "dollars": float(df_["u"].sum()),
                       "profit_per_dollar": float(df_["x"].sum() / df_["u"].sum()), "share_on_shift_side": float(df_["side"].sum() / df_["u"].sum())}
                if gname.startswith("Flagged") and len(df_) >= 5:
                    rg = np.random.default_rng(i)
                    ix = [rg.integers(0, len(df_), len(df_)) for _ in range(500)]
                    X, U, S = df_["x"].to_numpy(), df_["u"].to_numpy(), df_["side"].to_numpy()
                    pp = [X[j].sum() / U[j].sum() for j in ix]; ss = [S[j].sum() / U[j].sum() for j in ix]
                    row.update({"ppd_lo": np.percentile(pp, 2.5), "ppd_hi": np.percentile(pp, 97.5), "side_lo": np.percentile(ss, 2.5), "side_hi": np.percentile(ss, 97.5)})
                out.append(row)
        return out
    rb = pd.DataFrame(bucket_table("rel_bucket", "with_shift", REL_LABELS, "hourly") + bucket_table("rel10_bucket", "with_jump", FINE_LABELS, "10-minute"))
    rb.to_csv(tdir / "timing_by_shift_bucket.csv", index=False)

    # FOMC statement: minutes around the exact release time
    tf = t[t["is_fomc"] & t["hours_to_info"].notna()]
    fo_rows = []
    for gname, mask in groups.items():
        s = tf[mask[tf.index]]
        b = np.digitize(s["hours_to_info"].to_numpy(), FOMC_EDGES) - 1
        b = np.clip(b, 0, len(FOMC_LABELS) - 1)
        d = np.bincount(b, weights=s["usdc"].to_numpy(), minlength=len(FOMC_LABELS))
        for i, lab in enumerate(FOMC_LABELS[::-1]):
            fo_rows.append({"group": gname, "bucket": lab, "dollar_share": d[::-1][i] / max(d.sum(), 1e-9), "dollars": float(d[::-1][i]),
                            "wallets": int(s["wallet"].nunique())})
    fo = pd.DataFrame(fo_rows); fo.to_csv(tdir / "timing_fomc.csv", index=False)

    # headline numbers
    def pick(clock, group, bucket_list, col="wallet_equal_share"):
        d = shares_df[(shares_df.clock == clock) & (shares_df.group == group) & shares_df.bucket.isin(bucket_list)]
        return float(d[col].sum())
    F, O = "Flagged wallets (less conservative rule)", "Other experienced wallets"
    summ = {}
    for nm, clk, bl in (("first_10pct_of_life", "Share of the market's life elapsed", ["0-10%"]),
                        ("last_10pct_of_life", "Share of the market's life elapsed", ["90-100%"]),
                        ("last_24h_before_resolution", "Time before the market resolved", ["under 1 h", "1-6 h", "6-24 h"]),
                        ("more_than_30d_before_resolution", "Time before the market resolved", ["30-90 days", "90+ days"]),
                        ("within_24h_before_shift", "Time relative to the market's biggest price shift", ["7-24 h before", "2-6 h before", "the hour before"]),
                        ("hour_before_or_of_shift", "Time relative to the market's biggest price shift", ["the hour before", "the hour of the shift"]),
                        ("more_than_24h_before_shift", "Time relative to the market's biggest price shift", ["7+ days before", "1-7 days before"]),
                        ("after_shift", "Time relative to the market's biggest price shift", ["1-6 h after", "7-24 h after", "24+ h after"]),
                        ("ten_minutes_before_jump", "Minutes around the sharpest 10-minute jump", ["the 10 minutes before"]),
                        ("hour_before_jump", "Minutes around the sharpest 10-minute jump", ["30-60 min before", "10-30 min before", "the 10 minutes before"]),
                        ("ten_minutes_of_jump", "Minutes around the sharpest 10-minute jump", ["the 10 minutes of the jump"]),
                        ("hour_after_jump", "Minutes around the sharpest 10-minute jump", ["10-30 min after", "30-60 min after"])):
        summ[nm] = {"flagged_wallet_equal": pick(clk, F, bl), "other_wallet_equal": pick(clk, O, bl),
                    "flagged_pooled": pick(clk, F, bl, "pooled_dollar_share"), "other_pooled": pick(clk, O, bl, "pooled_dollar_share"),
                    "all_traders_pooled": pick(clk, "All traders", bl, "pooled_dollar_share")}
    summ["wallets"] = info["groups"]; summ["markets_with_shift"] = info["markets_with_shift"]; summ["markets"] = info["markets"]; summ["markets_with_sharp_jump"] = info.get("markets_with_sharp_jump")
    json.dump(summ, open(tdir / "timing_summary.json", "w"), indent=1)

    make_figures(shares_df, mo, rb, fo, fig_dir)
    print(json.dumps({k: v for k, v in summ.items() if isinstance(v, dict) and "flagged_wallet_equal" in v}, indent=1)[:2500])


def make_figures(shares_df, mo, rb, fo, fig_dir: Path, clocks_size=(13, 8.2), shift_size=(13, 8.4), suffix="", font=9):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": font, "axes.titlesize": font + 1, "axes.titleweight": "bold"})
    F, O = "Flagged wallets (less conservative rule)", "Other experienced wallets"
    cols = {F: AMBER, O: BLUE, "All traders": GREY}
    clocks = list(shares_df["clock"].unique())
    fig, ax = plt.subplots(2, 2, figsize=clocks_size); ax = ax.ravel()
    for a, c in zip(ax, clocks):
        d = shares_df[shares_df.clock == c]; labs = list(dict.fromkeys(d["bucket"])); x = np.arange(len(labs)); wd = 0.27
        for k, (g, colr) in enumerate(cols.items()):
            s = d[d.group == g].set_index("bucket").reindex(labs)
            col = "wallet_equal_share" if g != "All traders" else "pooled_dollar_share"
            y = s[col].to_numpy() * 100
            err = None
            if g == F:
                err = [np.maximum(y - s["wallet_equal_lo"].to_numpy() * 100, 0), np.maximum(s["wallet_equal_hi"].to_numpy() * 100 - y, 0)]
            a.bar(x + (k - 1) * wd, y, wd, color=colr, label=g + (" (each wallet counts once)" if g != "All traders" else " (dollars)"), yerr=err, capsize=1.5, error_kw={"lw": 0.8})
        a.set_xticks(x, labs, rotation=45, ha="right", fontsize=7); a.set_title(c, fontsize=9); a.set_ylabel("% of the group's dollars")
    ax[0].legend(frameon=False, fontsize=6.5)
    plt.tight_layout(); plt.savefig(fig_dir / f"timing_clocks{suffix}.png"); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 3.6)); hs = list(dict.fromkeys(mo["horizon"])); x = np.arange(len(hs)); wd = 0.36
    for k, g in enumerate((F, O)):
        s = mo[mo.group == g].set_index("horizon").reindex(hs); y = s["gain_per_dollar"].to_numpy() * 100
        err = [y - s["lo"].to_numpy() * 100, s["hi"].to_numpy() * 100 - y] if g == F else None
        ax.bar(x + (k - 0.5) * wd, y, wd, color=cols[g], label=g, yerr=err, capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x, hs); ax.set_xlabel("how long after the trade"); ax.set_ylabel("price gain on the held outcome (% per dollar)")
    ax.set_title("After they trade, does the price move their way?"); ax.legend(frameon=False, fontsize=7)
    plt.tight_layout(); plt.savefig(fig_dir / f"timing_markouts{suffix}.png"); plt.close()

    fig, ax = plt.subplots(2, 2, figsize=shift_size)
    for r_, (tag, labels) in enumerate((("hourly", REL_LABELS), ("10-minute", FINE_LABELS))):
        d_ = rb[rb.clock == tag]; labs = [l for l in labels if l in set(d_["bucket"])]; x = np.arange(len(labs))
        for k, g in enumerate((F, O)):
            s_ = d_[d_.group == g].set_index("bucket").reindex(labs)
            e1 = e2 = None
            if g == F:
                e1 = [np.maximum(s_["profit_per_dollar"] - s_["ppd_lo"], 0) * 100, np.maximum(s_["ppd_hi"] - s_["profit_per_dollar"], 0) * 100]
                e2 = [np.maximum(s_["share_on_shift_side"] - s_["side_lo"], 0) * 100, np.maximum(s_["side_hi"] - s_["share_on_shift_side"], 0) * 100]
            ax[r_, 0].bar(x + (k - 0.5) * 0.36, s_["profit_per_dollar"].to_numpy() * 100, 0.36, color=cols[g], label=g, yerr=e1, capsize=1.5, error_kw={"lw": 0.7})
            ax[r_, 1].bar(x + (k - 0.5) * 0.36, s_["share_on_shift_side"].to_numpy() * 100, 0.36, color=cols[g], label=g, yerr=e2, capsize=1.5, error_kw={"lw": 0.7})
        ax[r_, 0].axhline(0, color="k", lw=0.8); ax[r_, 1].axhline(50, color="k", lw=0.8, ls=":"); ax[r_, 0].set_ylim(-60, 130)
        nm = "biggest hourly shift" if tag == "hourly" else "sharpest 10-min jump"
        ax[r_, 0].set(title=f"Profit per dollar, by timing ({nm})", ylabel="% per dollar (axis cut at 130%)")
        ax[r_, 1].set(title="Share of dollars on the side the price moved to", ylabel="% of dollars", ylim=(0, 100))
        for a in ax[r_]: a.set_xticks(x, labs, rotation=40, ha="right", fontsize=7)
    ax[0, 0].legend(frameon=False, fontsize=7); plt.tight_layout(); plt.savefig(fig_dir / f"timing_by_shift{suffix}.png"); plt.close()

    if len(fo) and fo["dollars"].sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 3.5)); labs = list(dict.fromkeys(fo["bucket"])); x = np.arange(len(labs))
        for k, g in enumerate((F, O)):
            s = fo[fo.group == g].set_index("bucket").reindex(labs); ax.bar(x + (k - 0.5) * 0.36, s["dollar_share"] * 100, 0.36, color=cols[g], label=f"{g} ({int(s['wallets'].iloc[0])} wallets)")
        ax.set_xticks(x, labs, rotation=30, ha="right", fontsize=7); ax.set_ylabel("% of the group's dollars in Fed markets"); ax.set_title("Fed markets: trading around the statement time"); ax.legend(frameon=False, fontsize=7)
        plt.tight_layout(); plt.savefig(fig_dir / f"timing_fomc{suffix}.png"); plt.close()
