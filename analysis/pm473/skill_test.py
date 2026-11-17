"""Calibrated test for "likely informed" wallets (replaces the per-event ROI t-test).

Why a new test
--------------
The old test used each event's ROI. ROI is dominated by longshot wins (a 5-cent
contract that wins returns 1,900%), so the t-test's normal approximation fails,
and wallets that bet the same markets are not independent. On the crypto
placebo it flagged 4.5% of wallets; a correct test should flag almost none.

The test
--------
Null hypothesis: prices are fair, i.e. E[outcome | price at trade time] = price.
Under it, a wallet's profit on any fill is a mean-zero bet.

1. Per fill k: profit_k = dir0_k * size_k * (Y - p0_k), where Y = 1 if outcome 0
   won and p0_k is the fill price in outcome-0 units. Summed over a wallet's
   fills in a market (minus fees) this equals the PnL from pnl.py.
2. Null variance of the wallet's profit in one market. Because the price is a
   martingale, Cov(Y - p_j, Y - p_k) = E[p_m (1 - p_m)] for m = the later fill, so
        V = sum_j  p_j (1 - p_j) * n_j * (n_j + 2 * sum_{k<j} n_k),   n_j = dir0_j * size_j.
3. Event score z_e = (total profit in event) / sqrt(total V in event). Markets in
   the same event are summed (their true covariance is negative for "YES" bets
   on different outcomes, so this is conservative).
4. Wallet statistic Z = sum_e clip(z_e, -zcap, zcap) / sqrt(n_events)  (Stouffer).
   Clipping stops one lucky longshot from deciding a wallet's score.
5. Empirical null: the spread of Z across ALL eligible wallets is estimated
   robustly (median absolute deviation) and Z is divided by it if it exceeds 1.
   This absorbs whatever dependence the formula above misses. The centre is
   left at 0 (fair prices), which is the conservative choice because fees and
   spreads push the typical wallet below zero.
6. One-sided p-values, Benjamini-Hochberg FDR on the calibrated Z. Benjamini-
   Yekutieli (valid under any dependence) is available as method="BY" and is
   reported as a conservative sensitivity check: it needs roughly 10x smaller
   p-values and is very low-powered with thousands of wallets.

Validation lives in run_flagging_v2.py (placebo flag rate, planted-trader
recovery, noise-only simulation, out-of-sample persistence, sensitivity grid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

ZCAP = 3.0


# ------------------------------------------------------------ per fill -> per event
def wallet_market_terms(fills: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    """Profit and null variance per (wallet, market).

    fills needs: wallet, condition_id, ts, size, p0, dir0, usdc (+ fee_usdc).
    mk needs: condition_id, value_0 (1 if outcome 0 won, 0 if it lost)."""
    f = fills
    wallet = f["wallet"].to_numpy()
    cond = f["condition_id"].to_numpy()
    ts = f["ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    order = np.lexsort((ts, cond, wallet))
    wallet, cond = wallet[order], cond[order]
    size = f["size"].to_numpy(dtype="float64")[order]
    dir0 = f["dir0"].to_numpy(dtype="float64")[order]
    p0 = f["p0"].to_numpy(dtype="float64")[order]
    usdc = f["usdc"].to_numpy(dtype="float64")[order]
    fee = f["fee_usdc"].to_numpy(dtype="float64")[order] if "fee_usdc" in f else np.zeros(len(f))
    y = pd.Series(cond).map(mk.set_index("condition_id")["value_0"]).to_numpy(dtype="float64")

    n = dir0 * size
    new = np.r_[True, (wallet[1:] != wallet[:-1]) | (cond[1:] != cond[:-1])]
    gid = np.cumsum(new) - 1
    cs = np.cumsum(n)
    start = np.flatnonzero(new)
    prev = cs - n - (cs[start] - n[start])[gid]           # sum of earlier n within the group
    v = p0 * (1.0 - p0)
    var_term = v * n * (n + 2.0 * prev)
    profit = n * (y - p0) - fee

    k = len(start)
    out = pd.DataFrame({
        "wallet": wallet[start], "condition_id": cond[start],
        "c": np.bincount(gid, weights=np.nan_to_num(profit), minlength=k),
        "V": np.bincount(gid, weights=var_term, minlength=k),
        "volume": np.bincount(gid, weights=usdc, minlength=k),
        "n_fills": np.bincount(gid, minlength=k),
        "resolved": np.bincount(gid, weights=np.isfinite(y).astype(float), minlength=k) > 0,
    })
    return out[out["resolved"]].drop(columns="resolved").reset_index(drop=True)


def event_keys(mk: pd.DataFrame) -> pd.Series:
    """condition_id -> event key (outcomes of one multi-outcome event share a key)."""
    own = "m:" + mk["condition_id"].astype(str)
    if "event_id" in mk:
        ev = mk["event_id"]
        key = np.where(ev.notna(), "e:" + ev.astype(str), own)
    else:
        key = own
    return pd.Series(key, index=mk["condition_id"].to_numpy())


def period_times(mk: pd.DataFrame) -> pd.Series:
    """event_key -> time its LAST outcome resolves (decides formation vs evaluation)."""
    key = event_keys(mk)
    t = pd.Series(mk["resolve_time"].to_numpy(), index=key.reindex(mk["condition_id"]).to_numpy())
    return t.groupby(level=0).max()


def event_table(wm: pd.DataFrame, mk: pd.DataFrame, zcap: float = ZCAP) -> pd.DataFrame:
    """One row per (wallet, event): profit, null variance, volume, z, period time."""
    wm = wm.copy()
    wm["event_key"] = wm["condition_id"].map(event_keys(mk))
    ev = (wm.groupby(["wallet", "event_key"], sort=False)
            .agg(c=("c", "sum"), V=("V", "sum"), volume=("volume", "sum"), n_fills=("n_fills", "sum"))
            .reset_index())
    ok = ev["V"] > 1e-12
    ev["z"] = np.where(ok, ev["c"] / np.sqrt(ev["V"].where(ok, 1.0)), np.nan)
    ev["z_clipped"] = ev["z"].clip(-zcap, zcap)
    ev["period_time"] = ev["event_key"].map(period_times(mk))
    return ev


# ------------------------------------------------------------ wallet statistic + FDR
def wallet_stats(ev: pd.DataFrame, min_events: int = 10) -> pd.DataFrame:
    e = ev.dropna(subset=["z_clipped"])
    g = e.groupby("wallet")
    s = pd.DataFrame({"n_events": g.size(), "mean_z": g["z_clipped"].mean(),
                      "profit": g["c"].sum(), "volume": g["volume"].sum(), "V": g["V"].sum()})
    s["Z"] = g["z_clipped"].sum() / np.sqrt(s["n_events"])
    s["Z_dollar"] = s["profit"] / np.sqrt(s["V"])            # dollar-weighted cross-check
    s["profit_per_dollar"] = s["profit"] / s["volume"].replace(0, np.nan)
    s["eligible"] = s["n_events"] >= min_events
    return s


def empirical_sigma(z: np.ndarray, floor: float = 1.0) -> float:
    """Robust spread of the bulk of Z (MAD); never below `floor` (default: the theoretical 1)."""
    z = z[np.isfinite(z)]
    if len(z) < 30:
        return 1.0
    mad = np.median(np.abs(z - np.median(z))) * 1.4826
    return float(max(floor, mad))


def bh_by_mask(p: np.ndarray, q: float, method: str = "BY") -> np.ndarray:
    """Boolean mask of rejected hypotheses (BH = independent, BY = any dependence)."""
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    c = np.sum(1.0 / np.arange(1, m + 1)) if method == "BY" else 1.0
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / (m * c)
    k = np.max(np.flatnonzero(passed)) + 1 if passed.any() else 0
    mask = np.zeros(m, dtype=bool)
    mask[order[:k]] = True
    return mask


def flag_wallets(stats_df: pd.DataFrame, q: float = 0.10, method: str = "BH",
                 calibrate: bool = True, sigma_floor: float = 1.0, center: bool = False) -> pd.DataFrame:
    """Defaults reproduce the original rule (spread never below 1, centre fixed at 0).
    sigma_floor < 1 lets the empirical null be narrower than N(0,1); center=True also re-centres Z on the
    median of eligible wallets (Efron's empirical null). Both make the test less conservative."""
    s = stats_df.copy()
    el = s["eligible"].to_numpy()
    zel = s.loc[el, "Z"].to_numpy()
    sigma = empirical_sigma(zel, sigma_floor) if calibrate else 1.0
    mu = float(np.nanmedian(zel)) if (center and calibrate and el.any()) else 0.0
    s["Z_adj"] = (s["Z"] - mu) / sigma
    s["p"] = stats.norm.sf(s["Z_adj"])
    s["informed"] = False
    if el.any():
        s.loc[el, "informed"] = bh_by_mask(s.loc[el, "p"].to_numpy(), q, method)
    s.attrs["sigma"] = sigma
    s.attrs["center"] = mu
    return s


def calibration_report(s: pd.DataFrame) -> dict:
    """How well-behaved is Z across eligible wallets? Under a correct null,
    Z is ~N(0, 1) with a few real skilled wallets in the right tail."""
    z = s.loc[s["eligible"], "Z"].dropna().to_numpy()
    if not len(z):
        return {"eligible_wallets": 0}
    qs = [0.5, 0.9, 0.99]
    return {
        "eligible_wallets": int(len(z)),
        "sigma_hat": empirical_sigma(z),
        "sigma_hat_unfloored": empirical_sigma(z, floor=0.0),
        "sigma_used": s.attrs.get("sigma"), "centre_used": s.attrs.get("center", 0.0),
        "median_Z": float(np.median(z)),
        "share_Z_above_1.645": float((z > 1.645).mean()),       # 5% under a perfect null
        "share_Z_above_2.326": float((z > 2.326).mean()),       # 1% under a perfect null
        "quantiles_Z": {str(a): float(np.quantile(z, a)) for a in qs},
        "quantiles_normal": {str(a): float(stats.norm.ppf(a)) for a in qs},
        "flagged": int(s["informed"].sum()) if "informed" in s else None,
        "flag_rate": float(s.loc[s["eligible"], "informed"].mean()) if "informed" in s else None,
    }


# ------------------------------------------------------------ out-of-sample evaluation
def _boot(a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), (float("nan"), float("nan"))
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))


def evaluate(flagged: pd.DataFrame, ev_eval: pd.DataFrame, seed: int = 0) -> dict:
    """Do wallets flagged in formation still beat prices in the evaluation period?

    Primary metric: mean event z-score (0 under fair prices). Also reported:
    profit per dollar traded. Comparisons: all other eligible wallets, and a
    random wallet matched on formation volume and number of events."""
    rng = np.random.default_rng(seed)
    e = ev_eval.dropna(subset=["z_clipped"])
    g = e.groupby("wallet")
    t = pd.DataFrame({"eval_events": g.size(), "eval_mean_z": g["z_clipped"].mean(),
                      "eval_profit": g["c"].sum(), "eval_volume": g["volume"].sum()})
    t["eval_ppd"] = t["eval_profit"] / t["eval_volume"].replace(0, np.nan)
    p = flagged[flagged["eligible"]].join(t, how="inner")
    inf, rest = p[p["informed"]], p[~p["informed"]]
    out = {"n_informed_with_eval": int(len(inf)), "n_rest_with_eval": int(len(rest))}
    if len(inf) < 2 or len(rest) < 2:
        return out
    for col, name in (("eval_mean_z", "mean_z"), ("eval_ppd", "profit_per_dollar")):
        out[f"{name}_informed"] = float(inf[col].mean())
        out[f"{name}_rest"] = float(rest[col].mean())
        d, ci = _boot(inf[col], rest[col], seed=seed)
        out[f"{name}_diff_vs_all"], out[f"{name}_diff_vs_all_ci95"] = d, ci
    # matched comparison
    vb = pd.qcut(np.log1p(p["volume"]), q=min(10, p["volume"].nunique()), duplicates="drop", labels=False)
    nb = pd.qcut(p["n_events"].rank(method="first"), q=min(5, len(p)), duplicates="drop", labels=False)
    cell = pd.Series(vb.astype(int).astype(str) + "_" + nb.astype(int).astype(str), index=p.index)
    pool = {k: v.index.to_numpy() for k, v in rest.groupby(cell.loc[rest.index])}
    picks = [rng.choice(pool[cell[i]]) for i in inf.index if cell[i] in pool]
    if len(picks) > 1:
        m = p.loc[picks]
        for col, name in (("eval_mean_z", "mean_z"), ("eval_ppd", "profit_per_dollar")):
            d, ci = _boot(inf.loc[[i for i in inf.index if cell[i] in pool], col], m[col], seed=seed + 1)
            out[f"{name}_diff_vs_matched"], out[f"{name}_diff_vs_matched_ci95"] = d, ci
    # dollar-weighted (pooled) result for the informed group
    out["pooled_profit_per_dollar_informed"] = float(inf["eval_profit"].sum() / inf["eval_volume"].sum())
    out["pooled_profit_per_dollar_rest"] = float(rest["eval_profit"].sum() / rest["eval_volume"].sum())
    out["rank_corr_Z_vs_eval_mean_z"] = float(stats.spearmanr(p["Z"], p["eval_mean_z"])[0])
    return out


def run_split(ev: pd.DataFrame, split, min_events: int = 10, q: float = 0.10,
              method: str = "BH", calibrate: bool = True, sigma_floor: float = 1.0,
              center: bool = False) -> tuple[pd.DataFrame, dict]:
    """Flag on events resolving before `split`, evaluate on those after."""
    split = pd.Timestamp(split, tz="UTC") if pd.Timestamp(split).tzinfo is None else pd.Timestamp(split)
    form = ev[ev["period_time"] < split]
    evl = ev[ev["period_time"] >= split]
    s = flag_wallets(wallet_stats(form, min_events), q=q, method=method, calibrate=calibrate,
                     sigma_floor=sigma_floor, center=center)
    res = {"split": str(split.date()), "min_events": min_events, "q": q, "method": method,
           "calibration": calibration_report(s), "evaluation": evaluate(s, evl)}
    return s, res
