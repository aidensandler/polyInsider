"""'What if you just traded with the alerts?'  Historical copy-trading test.

An alert is a trade the model scored above a threshold. A follower would copy it with a flat stake:

  * ONE alert per (wallet, market, outcome held): the first qualifying trade. Later fills by the same
    wallet in the same market and outcome are not copied again.
  * The follower pays the price of the next trade in that market at or after (alert time + delay),
    expressed for the outcome the alerting wallet holds. With delay 0 the follower gets the alert
    trade's own price (an optimistic upper bound). We have no historical order books, so this is the
    best available execution price; an extra per-share cost (slippage) can be added.
  * Held to resolution: return = (payout - price paid - slippage) / (price paid + slippage).
  * Skipped: no later trade in the market, or a price paid below 0.02 or above 0.98.

Uncertainty: markets in one event are not independent, so confidence intervals resample whole EVENTS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LO, PRICE_HI = 0.02, 0.98


def _sec(ts: pd.Series) -> np.ndarray:
    """Unix seconds of a timestamp column (any resolution, tz-aware or not)."""
    s = ts.dt.tz_convert("UTC").dt.tz_localize(None) if getattr(ts.dt, "tz", None) is not None else ts
    return ((s - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).to_numpy("int64")


def dedupe_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first trade per (wallet, market, outcome held)."""
    d = df.sort_values("ts", kind="mergesort")
    return d.drop_duplicates(["wallet", "condition_id", "held_outcome"], keep="first")


def make_tape(path_or_df, condition_ids, since) -> pd.DataFrame:
    """All fills in the given markets from `since` on: condition_id, ts_sec, p0. Sorted for searchsorted."""
    if isinstance(path_or_df, pd.DataFrame):
        t = path_or_df[["condition_id", "ts", "p0"]]
        t = t[t["condition_id"].isin(set(condition_ids))]
    else:
        import pyarrow.parquet as pq
        t = pq.read_table(path_or_df, columns=["condition_id", "ts", "p0"],
                          filters=[("condition_id", "in", [int(c) for c in condition_ids])]).to_pandas()
    since = pd.Timestamp(since)
    since = since if since.tzinfo else since.tz_localize("UTC")
    t = t[t["ts"] >= since - pd.Timedelta(days=1)]
    out = pd.DataFrame({"condition_id": t["condition_id"].to_numpy(np.int64), "ts_sec": _sec(t["ts"]),
                        "p0": t["p0"].to_numpy("float64")})
    out["key"] = (out["condition_id"].to_numpy(np.int64) << 32) | out["ts_sec"].to_numpy(np.int64)
    return out.sort_values("key", kind="mergesort").reset_index(drop=True)


def exec_price(alerts: pd.DataFrame, tape: pd.DataFrame, delay_s: int) -> np.ndarray:
    """Price of the held outcome paid by a follower who acts `delay_s` seconds after the alert (NaN if none)."""
    if delay_s <= 0:
        return alerts["entry_price"].to_numpy("float64")
    a_sec = _sec(alerts["ts"])
    cond = alerts["condition_id"].to_numpy(np.int64)
    want = (cond << 32) | (a_sec + delay_s)
    idx = np.searchsorted(tape["key"].to_numpy(), want, side="left")
    ok = idx < len(tape)
    idx2 = np.where(ok, idx, 0)
    ok &= tape["condition_id"].to_numpy()[idx2] == cond
    p0 = tape["p0"].to_numpy()[idx2]
    held1 = alerts["held_outcome"].to_numpy() == 1
    p = np.where(held1, 1.0 - p0, p0)
    return np.where(ok, p, np.nan)


def follower_results(alerts: pd.DataFrame, tape: pd.DataFrame, delay_s: int = 0, stake: float = 100.0,
                     slippage: float = 0.0) -> pd.DataFrame:
    """One row per copied alert: price paid, return, PnL on a flat stake. Skipped alerts are dropped."""
    p = exec_price(alerts, tape, delay_s)
    r = alerts.assign(p_exec=p)
    r = r[r["p_exec"].between(PRICE_LO, PRICE_HI)].copy()
    cost = r["p_exec"] + slippage
    r["ret"] = (r["payout"] - cost) / cost
    r["pnl"] = stake * r["ret"]
    r["stake"] = stake
    return r


def _roi_ci(r: pd.DataFrame, n_boot: int, seed: int = 0):
    g = r.groupby("event_key")[["pnl", "stake"]].sum()
    idx = np.random.default_rng(seed).integers(0, len(g), (n_boot, len(g)))
    boot = g["pnl"].to_numpy()[idx].sum(1) / g["stake"].to_numpy()[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(r["pnl"].sum() / r["stake"].sum()), float(lo), float(hi), len(g)


def summarize(r: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    """Totals, return per dollar staked with a 95% interval from resampling events, win rate, and how
    concentrated the result is: the single wallet that supplied the most profit, its share of total profit,
    and the return without that wallet."""
    if r.empty:
        return {"alerts": 0}
    roi, lo, hi, n_events = _roi_ci(r, n_boot, seed)
    by_w = r.groupby("wallet")["pnl"].agg(["size", "sum"])
    top = by_w["sum"].idxmax(); total = by_w["sum"].sum()
    out = {"alerts": int(len(r)), "events": int(n_events), "wallets": int(r["wallet"].nunique()),
           "stake_total": float(r["stake"].sum()), "pnl_total": float(r["pnl"].sum()),
           "roi": roi, "roi_lo": lo, "roi_hi": hi,
           "win_rate": float((r["payout"] > 0.5).mean()), "median_return": float(r["ret"].median()),
           "mean_price_paid": float(r["p_exec"].mean()),
           "top_wallet_alert_share": float(by_w["size"].max() / by_w["size"].sum()),
           "top_profit_wallet_share": float(by_w["sum"].max() / total) if total > 0 else np.nan}
    rest = r[r["wallet"] != top]
    if len(rest) >= 20 and rest["event_key"].nunique() >= 3:
        a, b, c, _ = _roi_ci(rest, n_boot, seed)
        out.update({"roi_without_top_wallet": a, "roi_wo_lo": b, "roi_wo_hi": c, "alerts_without_top_wallet": int(len(rest))})
    return out


def random_benchmark(pool: pd.DataFrame, tape: pd.DataFrame, n_alerts: int, delay_s: int, slippage: float = 0.0,
                     n_draws: int = 300, seed: int = 0) -> dict:
    """Return per dollar of the same number of randomly chosen (deduped) candidate trades, many times.
    Same copying rules, so the comparison is fair. Reports mean and the 5th to 95th percentile."""
    base = follower_results(pool, tape, delay_s, slippage=slippage)
    if len(base) < n_alerts or n_alerts == 0:
        return {}
    rng = np.random.default_rng(seed)
    pnl, stake = base["pnl"].to_numpy(), base["stake"].to_numpy()
    rois = np.array([pnl[i].sum() / stake[i].sum() for i in (rng.choice(len(base), n_alerts, replace=False) for _ in range(n_draws))])
    return {"mean": float(rois.mean()), "p05": float(np.percentile(rois, 5)), "p95": float(np.percentile(rois, 95)),
            "draws": rois}


def pnl_curve(r: pd.DataFrame, event_end: pd.Series) -> pd.DataFrame:
    """Cumulative PnL, cumulative stake and cumulative return per dollar, by the day the event resolved
    (when the profit is realised)."""
    d = r.assign(day=r["event_key"].map(event_end).dt.floor("D")).groupby("day")[["pnl", "stake"]].sum().sort_index().cumsum()
    d = d.rename(columns={"pnl": "cum_pnl", "stake": "cum_stake"})
    d["cum_roi"] = d["cum_pnl"] / d["cum_stake"]
    return d.reset_index()
