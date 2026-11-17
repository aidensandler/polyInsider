"""Alert model: score each trade, in a market that is still open, by how much the trade is
expected to earn per share when held to resolution.

Everything a trade is scored with is known AT THE TIME OF THE TRADE:
  * the wallet's history = only the events that had already resolved before the trade
    (strictly earlier `period_time`),
  * the trade itself (price paid for the outcome it holds, dollars, side),
  * how long until the market's SCHEDULED end or release time (known in advance). Not the realised
    resolve time, and not the cleaned data's `hours_to_info`, which falls back to the realised resolve time
    for markets that resolved before their scheduled end.

Label (only for training and evaluation, never an input):
  edge = payout - entry_price   per share of the outcome the trade holds
  (payout is 1 if that outcome wins, else 0). At fair prices its average is 0.

Train on trades made before the cut-off in events that had resolved by the cut-off.
Score trades made after it. See `train_alert_model` and `score`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["entry_price", "held_outcome", "log_usdc", "log_hours_to_sched", "tag_code",
            "n_past", "Z_past", "mean_z_past", "ppd_past", "win_rate_past", "log_vol_past", "last_z"]
WALLET_FEATURES = ["n_past", "Z_past", "mean_z_past", "ppd_past", "win_rate_past", "log_vol_past", "last_z"]
TRADE_FEATURES = ["entry_price", "held_outcome", "log_usdc", "log_hours_to_sched", "tag_code"]
PRICE_FEATURES = ["entry_price", "held_outcome"]
TAGS = ["elections", "economy", "fed"]            # everything else -> 3
FILL_COLS = ["wallet", "condition_id", "ts", "dir0", "size", "p0", "usdc", "hours_to_resolve"]


# ----------------------------------------------------------------------------- wallet history
def wallet_history(ev: pd.DataFrame) -> pd.DataFrame:
    """Cumulative per-wallet statistics, one row per (wallet, event), stamped with the event's
    resolution time. Row k describes the wallet's first k events (in resolution order)."""
    e = ev[["wallet", "period_time", "c", "volume", "z_clipped"]].copy()
    e = e.sort_values(["wallet", "period_time"], kind="mergesort")
    valid = e["z_clipped"].notna()
    e["_z"] = e["z_clipped"].fillna(0.0)
    e["_v"] = valid.astype("int32")
    e["_w"] = ((e["c"] > 0) & valid).astype("int32")
    g = e.groupby("wallet", sort=False)
    out = pd.DataFrame({
        "wallet": e["wallet"].to_numpy(), "hist_time": e["period_time"].to_numpy(),
        "n_past": g["_v"].cumsum().to_numpy(), "sum_z": g["_z"].cumsum().to_numpy(),
        "sum_c": g["c"].cumsum().to_numpy(), "sum_vol": g["volume"].cumsum().to_numpy(),
        "wins": g["_w"].cumsum().to_numpy(), "last_z": e["_z"].to_numpy()})
    return out.sort_values("hist_time", kind="mergesort").reset_index(drop=True)


def add_wallet_features(trades: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Attach the wallet's history as of each trade's timestamp (events resolved strictly before it).
    `trades` needs wallet and ts. Wallets with no resolved event yet get n_past = 0 and NaN features."""
    t = trades.sort_values("ts", kind="mergesort").copy()
    h = hist[hist["wallet"].isin(t["wallet"].unique())]
    ts_dtype = t["ts"].dtype
    h = h.assign(hist_time=h["hist_time"].astype(ts_dtype))
    m = pd.merge_asof(t, h, left_on="ts", right_on="hist_time", by="wallet", allow_exact_matches=False)
    n = m["n_past"].fillna(0)
    m["n_past"] = n
    pos = n > 0
    sqrt_n = np.sqrt(n.where(pos, np.nan))
    m["Z_past"] = m["sum_z"] / sqrt_n
    m["mean_z_past"] = m["sum_z"] / n.where(pos, np.nan)
    m["ppd_past"] = m["sum_c"] / m["sum_vol"].where(m["sum_vol"] > 0, np.nan)
    m["win_rate_past"] = m["wins"] / n.where(pos, np.nan)
    m["log_vol_past"] = np.log1p(m["sum_vol"].fillna(0))
    return m.drop(columns=["sum_z", "sum_c", "sum_vol", "wins", "hist_time"])


# ----------------------------------------------------------------------------- trade table
def scheduled_times(mk: pd.DataFrame) -> pd.Series:
    """Scheduled end or release time per market (index = condition_id), known before the market resolves.
    Fed markets use the FOMC statement time, event markets their start time, the rest their end date."""
    sched = mk["info_time"].where(mk["info_time_source"] != "scheduled_end", mk["end_time"]).fillna(mk["info_time"])
    return pd.Series(sched.to_numpy(), index=mk["condition_id"].to_numpy())


def tag_codes(mk: pd.DataFrame) -> pd.Series:
    code = {t: i for i, t in enumerate(TAGS)}
    return mk.set_index("condition_id")["tag"].map(lambda t: code.get(t, len(TAGS))).astype("int8")


def trade_features(fills: pd.DataFrame, mk: pd.DataFrame, ek: pd.Series, pt: pd.Series, hist: pd.DataFrame,
                   min_past: int = 3, min_usd: float = 5.0) -> pd.DataFrame:
    """One row per fill with causal features, the held-outcome entry price, and (for training and
    evaluation only) the realised edge. Keeps fills with n_past >= min_past and usdc >= min_usd."""
    f = fills[fills["usdc"] >= min_usd].copy()
    if f.empty:
        return f
    f = add_wallet_features(f, hist)
    f = f[f["n_past"] >= min_past].copy()
    if f.empty:
        return f
    cm = mk.set_index("condition_id")
    long0 = f["dir0"].to_numpy() > 0
    p0 = f["p0"].to_numpy("float64")
    y = f["condition_id"].map(cm["value_0"]).to_numpy("float64")
    f["held_outcome"] = np.where(long0, 0, 1).astype("int8")
    f["entry_price"] = np.where(long0, p0, 1.0 - p0)
    f["payout"] = np.where(long0, y, 1.0 - y)
    f["edge"] = f["payout"] - f["entry_price"]
    f["log_usdc"] = np.log1p(f["usdc"].astype("float64"))
    sched = f["condition_id"].map(scheduled_times(mk))
    f["hours_to_sched"] = ((sched - f["ts"]).dt.total_seconds() / 3600.0)
    f["log_hours_to_sched"] = np.log1p(f["hours_to_sched"].clip(lower=0).astype("float64"))
    f["tag_code"] = f["condition_id"].map(tag_codes(mk)).astype("int8")
    f["event_key"] = f["condition_id"].map(ek)
    f["event_time"] = f["event_key"].map(pt)
    keep = ["wallet", "condition_id", "event_key", "ts", "event_time", "hours_to_resolve", "size", "usdc",
            "payout", "edge"] + FEATURES
    f = f[keep]
    num = f.select_dtypes("float64").columns
    f[num] = f[num].astype("float32")
    return f.reset_index(drop=True)


def build_candidates_from_parquet(path, mk: pd.DataFrame, ev: pd.DataFrame, min_past: int = 3,
                                  min_usd: float = 5.0, groups_per_chunk: int = 60, progress=None) -> pd.DataFrame:
    """Read a cleaned fills parquet a few row groups at a time (low memory) and build the trade table."""
    import pyarrow.parquet as pq
    from pm473 import skill_test as st
    ek, pt = st.event_keys(mk), st.period_times(mk)
    hist = wallet_history(ev)
    pf = pq.ParquetFile(path)
    parts, n_groups = [], pf.metadata.num_row_groups
    for i in range(0, n_groups, groups_per_chunk):
        tbl = pf.read_row_groups(list(range(i, min(n_groups, i + groups_per_chunk))), columns=FILL_COLS)
        parts.append(trade_features(tbl.to_pandas(), mk, ek, pt, hist, min_past, min_usd))
        if progress:
            progress(i, n_groups)
    return pd.concat(parts, ignore_index=True)


# ----------------------------------------------------------------------------- context features
# Ideas 1-3, added AFTER the candidate table is built, using only earlier trades in the table.
BOT_FEATURES = ["lw_trades_past", "lw_two_sided_past", "lw_trades_per_mkt_past"]
BEHAVIOUR_FEATURES = ["lw_mean_logusd_past", "rel_size", "lw_mean_hours_past", "lw_near24_past", "lw_mean_price_past",
                      "pos_n_same", "pos_log_usd_same", "opp_in_mkt"]
AGREE_FEATURES = ["is_good", "agree_log_usd_all", "agree_n_1h", "agree_log_usd_24h", "oppose_log_usd_24h", "agree_share_24h"]
CONTEXT_FEATURES = BOT_FEATURES + BEHAVIOUR_FEATURES + AGREE_FEATURES
# crowd flow: dollars from ALL qualifying wallets (no skill filter) on the same / other side of the market; the control
# that tells 'good wallets agree' apart from 'money is flowing into the favourite'
FLOW_FEATURES = ["flow_log_usd_all", "flow_log_usd_24h", "flow_opp_log_usd_24h", "flow_share_24h"]
_B = np.int64(1) << 32


def _window_sum(values, key_src, ts_src, key_q, ts_q, window_s=None):
    """For each query (key_q, ts_q): sum of `values` over source rows with the same key and a timestamp in
    [ts_q - window_s, ts_q), i.e. strictly earlier (window_s None = all earlier). Timestamps are whole seconds."""
    order = np.lexsort((ts_src, key_src))
    comp = key_src[order].astype(np.int64) * _B + ts_src[order].astype(np.int64)
    cum = np.concatenate([[0.0], np.cumsum(values[order].astype("float64"))])
    hi = np.searchsorted(comp, key_q.astype(np.int64) * _B + ts_q.astype(np.int64), side="left")
    start = ts_q.astype(np.int64) - (window_s if window_s is not None else ts_q.astype(np.int64))
    lo = np.searchsorted(comp, key_q.astype(np.int64) * _B + np.maximum(start, 0), side="left")
    return cum[hi] - cum[lo]


def add_context_features(c: pd.DataFrame, good_z: float = 2.0, good_n: int = 5) -> pd.DataFrame:
    """Add bot-likeness, behaviour and agreement features to a candidate table (all causal).

    Every feature at trade i uses only trades in the table with an EARLIER timestamp (same wallet for
    wallet-level features, same market for agreement features; trades in the same second are excluded from
    other wallets' agreement counts). No outcome or label is used. Because the table keeps only trades of at
    least $5 by wallets with 3+ resolved events, 'past' means 'past qualifying trades'.
    Returns the table sorted by ts with the new columns (float32).
    """
    c = c.sort_values("ts", kind="mergesort").reset_index(drop=True)
    n = len(c)
    ts_s = (c["ts"].to_numpy("datetime64[ns]").astype("int64") // 10**9)
    wal = c["wallet"].to_numpy("int64")
    cond = c["condition_id"].to_numpy("int64")
    held = c["held_outcome"].to_numpy("int64")
    usd = c["usdc"].to_numpy("float64")
    class _F32(dict):                                    # keep memory low: store every feature as float32
        def __setitem__(self, k, v):
            super().__setitem__(k, np.asarray(v, dtype="float32"))
    f = _F32()

    # --- wallet-level history of qualifying trades
    gw = c.groupby("wallet", sort=False)
    cnt = gw.cumcount().to_numpy("float64")                  # earlier trades by this wallet
    has = cnt > 0
    safe = np.where(has, cnt, np.nan)

    def prior_mean(x):
        x = np.asarray(x, dtype="float64")
        cs = pd.Series(x).groupby(wal, sort=False).cumsum().to_numpy()
        return (cs - x) / safe

    # --- same wallet, same market (and same outcome)
    key_wm = pd.factorize(wal * 100_000 + cond)[0]
    key_wmo = pd.factorize((wal * 100_000 + cond) * 2 + held)[0]
    n_mkt_prior = pd.Series(np.zeros(n)).groupby(key_wm).cumcount().to_numpy("float64")
    n_same_prior = pd.Series(np.zeros(n)).groupby(key_wmo).cumcount().to_numpy("float64")
    usd_same_prior = pd.Series(usd).groupby(key_wmo).cumsum().to_numpy() - usd
    opp_in = ((n_mkt_prior - n_same_prior) > 0).astype("float64")        # already held the other outcome here
    first_in_mkt = (n_mkt_prior == 0).astype("float64")
    distinct_prior = pd.Series(first_in_mkt).groupby(wal, sort=False).cumsum().to_numpy() - first_in_mkt
    f["lw_trades_past"] = np.log1p(cnt)
    f["lw_two_sided_past"] = prior_mean(opp_in)
    f["lw_trades_per_mkt_past"] = np.where(distinct_prior > 0, cnt / np.where(distinct_prior > 0, distinct_prior, 1), np.nan)

    lu = c["log_usdc"].to_numpy("float64")
    f["lw_mean_logusd_past"] = prior_mean(lu)
    f["rel_size"] = lu - f["lw_mean_logusd_past"]
    f["lw_mean_hours_past"] = prior_mean(c["log_hours_to_sched"].to_numpy("float64"))
    f["lw_near24_past"] = prior_mean((c["log_hours_to_sched"].to_numpy("float64") <= np.log1p(24.0)).astype("float64"))
    f["lw_mean_price_past"] = prior_mean(c["entry_price"].to_numpy("float64"))
    f["pos_n_same"] = n_same_prior
    f["pos_log_usd_same"] = np.log1p(usd_same_prior)
    f["opp_in_mkt"] = opp_in

    # --- agreement: earlier trades by OTHER 'good' wallets in this market
    good = ((c["Z_past"].to_numpy("float64") >= good_z) & (c["n_past"].to_numpy("float64") >= good_n))
    g_usd, g_one = usd * good, good.astype("float64")
    key_mo = cond * 2 + held
    key_mo_opp = cond * 2 + (1 - held)
    # own earlier good trades in the same (wallet, market, outcome) / opposite outcome, to leave them out
    wmo_src = (wal * 100_000 + cond) * 2 + held
    wmo_opp = (wal * 100_000 + cond) * 2 + (1 - held)

    def others(values, window, opposite=False):
        kq = key_mo_opp if opposite else key_mo
        tot = _window_sum(values, key_mo, ts_s, kq, ts_s, window)
        own = _window_sum(values, wmo_src, ts_s, wmo_opp if opposite else wmo_src, ts_s, window)
        return np.maximum(tot - own, 0.0)

    a_all = others(g_usd, None)
    a_1h_n = others(g_one, 3600)
    a_24 = others(g_usd, 86400)
    o_24 = others(g_usd, 86400, opposite=True)
    f["is_good"] = good.astype("float64")
    f["agree_log_usd_all"] = np.log1p(a_all)
    f["agree_n_1h"] = a_1h_n
    f["agree_log_usd_24h"] = np.log1p(a_24)
    f["oppose_log_usd_24h"] = np.log1p(o_24)
    f["agree_share_24h"] = np.where(a_24 + o_24 > 0, a_24 / np.where(a_24 + o_24 > 0, a_24 + o_24, 1), np.nan)
    # crowd flow: same computation without the 'good wallet' filter
    del a_all, a_1h_n, a_24, o_24
    fa_all, fa_24, fo_24 = others(usd, None), others(usd, 86400), others(usd, 86400, opposite=True)
    f["flow_log_usd_all"] = np.log1p(fa_all)
    f["flow_log_usd_24h"] = np.log1p(fa_24)
    f["flow_opp_log_usd_24h"] = np.log1p(fo_24)
    f["flow_share_24h"] = np.where(fa_24 + fo_24 > 0, fa_24 / np.where(fa_24 + fo_24 > 0, fa_24 + fo_24, 1), np.nan)
    return pd.concat([c, pd.DataFrame(f, index=c.index)], axis=1)


# ----------------------------------------------------------------------------- model
def split_train_test(c: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: trades made before the cut-off in events that had resolved by the cut-off (labels known then).
    Test: trades made at or after the cut-off."""
    cut = pd.Timestamp(cutoff)
    cut = cut if cut.tzinfo else cut.tz_localize("UTC")
    ts = c["ts"]
    return c[(ts < cut) & (c["event_time"] < cut)], c[ts >= cut]


def _model(max_iter: int, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=max_iter, max_depth=4,
                                         min_samples_leaf=2000, l2_regularization=1.0, early_stopping=False,
                                         random_state=seed)


def train_alert_model(train: pd.DataFrame, features=FEATURES, max_train_rows: int = 1_500_000, seed: int = 0,
                      iter_grid=(40, 100, 200), alert_quantile: float = 0.99):
    """Fit on `train` (never sees test). The number of boosting rounds is chosen on the LAST 20% of the
    training period (by time), then the model is refit on all of it. The alert threshold is the
    `alert_quantile` of scores on that last 20% (from the model fitted without it), so it is fixed before
    any test trade is seen. Returns (model, info); info["threshold"] is the alert cut-off."""
    from scipy.stats import spearmanr
    tr = train.sort_values("ts", kind="mergesort")
    if len(tr) > max_train_rows:
        tr = tr.sample(max_train_rows, random_state=seed).sort_values("ts", kind="mergesort")
    k = int(len(tr) * 0.8)
    a, b = tr.iloc[:k], tr.iloc[k:]
    best, best_rho, best_model, scores = iter_grid[0], -np.inf, None, {}
    for it in iter_grid:
        m = _model(it, seed).fit(a[features], a["edge"])
        rho = spearmanr(m.predict(b[features]), b["edge"]).correlation
        scores[it] = float(rho)
        if rho > best_rho:
            best, best_rho, best_model = it, rho, m
    threshold = float(np.quantile(best_model.predict(b[features]), alert_quantile))
    final = _model(best, seed).fit(tr[features], tr["edge"])
    return final, {"rounds": best, "validation_rank_corr": scores, "train_rows": int(len(tr)), "threshold": threshold,
                   "alert_quantile": alert_quantile, "validation_cut_time": str(b["ts"].iloc[0]) if len(b) else None}


def score(model, df: pd.DataFrame, features=FEATURES) -> np.ndarray:
    """Predicted edge per share (cents per share / 100)."""
    return model.predict(df[features])
