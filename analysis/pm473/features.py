"""Step 4: features describing *how* each wallet trades.

These map onto the research question:
  * markouts           -> does price move toward the wallet's position after it trades? (Kyle-style informed flow)
  * timing             -> does it wait until close to resolution before acting? (Chen et al. delayed revelation)
  * order splitting    -> does it break a position into many child orders over a short window?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = {"5m": "5min", "1h": "1h", "24h": "24h"}
RUN_GAP = pd.Timedelta("10min")


def _ns(s: pd.Series) -> np.ndarray:
    """UTC epoch nanoseconds as int64 (works on pandas 2.x and 3.x)."""
    s = pd.to_datetime(s, utc=True).dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]").view("int64")


def add_markouts(fills: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Per-fill markouts in outcome-0 probability points:
        markout_h = dir0 * (P0(t+h) - p0_fill)
    P0 is the last traded price (from both outcome tapes) at or before t+h.
    Positive = the market later moved in the trader's favour.

    Works market by market on numpy arrays (binary search on each market's
    time-sorted tape) and adds columns to `fills` in place, so extra memory
    is a few numeric arrays rather than copies of the table."""
    n = len(fills)
    codes, _ = pd.factorize(fills["condition_id"])
    t = _ns(fills["ts"])
    res = _ns(fills["resolve_time"])
    p0 = fills["p0"].to_numpy(dtype=float)
    d0 = fills["dir0"].to_numpy(dtype=float)
    order = np.lexsort((t, codes))                 # by market, then time
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)
    del order, bounds
    for name, delta in HORIZONS.items():
        dt = pd.Timedelta(delta).value
        out = np.full(n, np.nan, dtype="float32")
        for g in groups:
            tg, pg = t[g], p0[g]
            target = tg + dt
            j = np.searchsorted(tg, target, side="right") - 1   # last tape point <= t+h
            fut = pg[j]
            fut = np.where(target <= res[g], fut, np.nan)       # never look past resolution
            out[g] = d0[g] * (fut - pg)
        fills[f"mk_{name}"] = out
    v0 = fills["condition_id"].map(markets.set_index("condition_id")["value_0"]).to_numpy(dtype=float)
    fills["mk_terminal"] = (d0 * (v0 - p0)).astype("float32")
    return fills


def child_orders(fills: pd.DataFrame) -> pd.DataFrame:
    """One taker order sweeping several makers shows up as several fills in the
    SAME transaction -- that is not the trader splitting. Collapse to one row per
    (wallet, market, tx, direction) before measuring splitting."""
    return (fills.groupby(["wallet", "condition_id", "tx_hash", "dir0"], dropna=False)
            .agg(ts=("ts", "min"), size=("size", "sum"), usdc=("usdc", "sum"))
            .reset_index().sort_values(["wallet", "condition_id", "dir0", "ts"]))


def split_runs(orders: pd.DataFrame) -> pd.DataFrame:
    o = orders.copy()
    grp = ["wallet", "condition_id", "dir0"]
    gap = o.groupby(grp)["ts"].diff()
    o["run"] = (gap.isna() | (gap > RUN_GAP)).groupby([o[c] for c in grp]).cumsum()
    runs = o.groupby(grp + ["run"]).agg(n_orders=("ts", "size"), usdc=("usdc", "sum"),
                                         t0=("ts", "min"), t1=("ts", "max")).reset_index()
    return runs


def wallet_features(fills: pd.DataFrame) -> pd.DataFrame:
    """fills must already have markouts (add_markouts)."""
    f = fills
    w = f["size"]

    def wavg(col):
        x = f[col]
        ok = x.notna()
        num = (x.where(ok, 0) * w).groupby(f["wallet"]).sum()
        den = w.where(ok, 0).groupby(f["wallet"]).sum()
        return num / den.replace(0, np.nan)

    g = f.groupby("wallet")
    feat = pd.DataFrame({
        "n_markets": g["condition_id"].nunique(),
        "n_fills": g.size(),
        "volume_usdc": g["usdc"].sum(),
        "log_volume": np.log1p(g["usdc"].sum()),
        "avg_fill_usdc": g["usdc"].mean(),
        "frac_life_vw": wavg("frac_life"),
        "share_vol_last24h": (f["usdc"].where(f["hours_to_resolve"] <= 24, 0)
                              .groupby(f["wallet"]).sum() / g["usdc"].sum()),
        "avg_entry_price_extremity": (f["p0"] - 0.5).abs().groupby(f["wallet"]).mean(),
    })
    if "hours_to_info" in f:
        h = f["hours_to_info"]
        tot = g["usdc"].sum()
        # Chen et al.: informed traders may hold back, then trade just before the news
        feat["share_vol_24h_pre_info"] = f["usdc"].where((h > 0) & (h <= 24), 0).groupby(f["wallet"]).sum() / tot
        feat["share_vol_7d_pre_info"] = f["usdc"].where((h > 0) & (h <= 168), 0).groupby(f["wallet"]).sum() / tot
        feat["share_vol_post_info"] = f["usdc"].where(h <= 0, 0).groupby(f["wallet"]).sum() / tot
        pre = f[h > 0]
        feat["median_hours_before_info"] = pre.groupby("wallet")["hours_to_info"].median()
    for name in list(HORIZONS) + ["terminal"]:
        feat[f"mk_{name}"] = wavg(f"mk_{name}")

    # share of each market's volume this wallet accounts for (size / "whale-ness")
    mvol = f.groupby("condition_id")["usdc"].transform("sum")
    feat["mean_mkt_share"] = (f["usdc"] / mvol).groupby([f["wallet"], f["condition_id"]]).sum() \
        .groupby("wallet").mean()

    # maker/taker, if the v2 feed exposes it
    if "role" in f and f["role"].notna().any():
        r = f["role"].astype(str).str.lower()
        is_maker = r.str.contains("maker") | r.isin(["false", "0"])
        feat["maker_share"] = (f["usdc"].where(is_maker, 0).groupby(f["wallet"]).sum()
                               / g["usdc"].sum())

    # order splitting
    runs = split_runs(child_orders(f))
    rg = runs.groupby("wallet")
    feat["orders_per_run"] = rg["n_orders"].mean()
    feat["share_vol_split_runs"] = (runs["usdc"].where(runs["n_orders"] >= 3, 0)
                                    .groupby(runs["wallet"]).sum() / rg["usdc"].sum())
    return feat.fillna({"maker_share": 0.5}) if "maker_share" in feat else feat
