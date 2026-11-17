"""When do likely-informed wallets trade? (timing analysis)

    python timing_analysis.py process      # one pass over the evaluation-period markets (a few minutes); writes results_v4/timing/
    python timing_analysis.py report       # tables + figures from the saved per-trade table

Who: wallets flagged BEFORE the cut-off (default 2026-06-01) under the less conservative rule (R2) and the original
rule (R0), compared with the other experienced wallets (at least 10 events before the cut-off, not flagged) and with
all traders. Only events that resolved AFTER the cut-off are used, so the flag was never based on these trades:
if flagged wallets trade at special moments here, it is not because they were selected for it.

Four clocks (all computed from the trade tape):
  1. fraction of the market's life elapsed (0 = opened, 1 = resolved)       -> "at the beginning vs near the end"
  2. hours before the market resolved                                       -> "how long before resolution"
  3. hours relative to the market's BIGGEST PRICE SHIFT (the revelation proxy): the hour with the largest change in the
     hourly volume-weighted price (at least 15 cents, between active hours at most 6 h apart; hours with under $100 ignored)
                                                                           -> "right before / during / after the price moved"
  4. minutes relative to the SHARPEST 10-MINUTE JUMP (at least 10 cents between active 10-minute bins at most 30 min apart)
                                                                           -> "the minutes right before the news hits"
  5. minutes around the FOMC statement time for Fed markets (exact times known).
Plus markouts: how far the price moved in the trade's favour 5 min, 1 h, 6 h, 24 h later and at resolution, and where the
profit is earned. Everything is descriptive: a wallet that trades right before price shifts looks informed, but a fast
reaction to public news looks the same only if it comes BEFORE the move; trades after the move are not informed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pm473 import skill_test as st

SPLIT = "2026-06-01"
SHIFT_MIN, GAP_MAX_H, HOUR_MIN_USD = 0.15, 6, 100.0
REL_EDGES = [-168, -24, -6, -1, 0, 1, 7, 25]                # bucket edges on relative hours (right-open)
REL_LABELS = ["more than 7 days before", "1 to 7 days before", "7 to 24 hours before", "2 to 6 hours before",
              "the hour before", "the hour of the shift", "1 to 6 hours after", "7 to 24 hours after", "more than 24 hours after"]
FINE_EDGES = [-360, -60, -30, -10, 0, 10, 30, 60, 360]       # minutes relative to the sharp-shift 10-minute bin (right-open)
FINE_LABELS = ["6+ h before", "1-6 h before", "30-60 min before", "10-30 min before", "the 10 minutes before", "the 10 minutes of the jump",
               "10-30 min after", "30-60 min after", "1-6 h after", "6+ h after"]
FINE_MIN, FINE_GAP_BINS, FINE_BIN_USD = 0.10, 3, 50.0
HORIZONS = {"m5m": 300, "m1h": 3600, "m6h": 6 * 3600, "m24h": 24 * 3600}
COLS = ["wallet", "condition_id", "timestamp", "p0", "usdc", "dir0", "frac_life", "hours_to_resolve", "hours_to_info"]
GROUP_BITS = {"Experienced (10+ events before the cut-off)": 1, "Flagged (less conservative rule)": 2, "Flagged (original rule)": 4}


def groups_for(ev: pd.DataFrame, split: pd.Timestamp, n_codes: int):
    """Wallet-code -> bit flags: 1 = experienced (10+ events before the cut-off), 2 = flagged by the less conservative
    rule (R2), 4 = flagged by the original rule (R0). Only formation data (events resolved before the cut-off) is used."""
    form = ev[ev["period_time"] < split]
    stats_df = st.wallet_stats(form, 10)
    s0 = st.flag_wallets(stats_df, q=0.10, method="BH", sigma_floor=1.0, center=False)
    s2 = st.flag_wallets(stats_df, q=0.10, method="BH", sigma_floor=0.0, center=True)
    g = np.zeros(n_codes, dtype=np.int8)
    g[s2.index[s2["eligible"]].to_numpy()] |= 1
    g[s2.index[s2["informed"]].to_numpy()] |= 2
    g[s0.index[s0["informed"]].to_numpy()] |= 4
    return g


def market_shift(t_sec: np.ndarray, p0: np.ndarray, usdc: np.ndarray):
    """(hour_of_biggest_shift, signed size in outcome-0 price) or (nan, 0). t_sec sorted ascending."""
    hour = t_sec // 3600
    brk = np.flatnonzero(np.r_[True, hour[1:] != hour[:-1]])
    hr = hour[brk]
    vol = np.add.reduceat(usdc, brk)
    vw = np.add.reduceat(p0 * usdc, brk) / np.maximum(vol, 1e-9)
    ok = vol >= HOUR_MIN_USD
    hr, vw = hr[ok], vw[ok]
    if len(hr) < 3:
        return np.nan, 0.0
    d = np.diff(vw)
    gap = np.diff(hr)
    d = np.where(gap <= GAP_MAX_H, d, 0.0)
    k = int(np.argmax(np.abs(d)))
    return float(hr[k + 1]), float(d[k])


def market_jump(t_sec: np.ndarray, p0: np.ndarray, usdc: np.ndarray):
    """The sharpest move: largest change in the 10-minute volume-weighted price between active 10-minute bins at most
    30 minutes apart (bins under $50 ignored). Returns (bin index, signed size in outcome-0 price) or (nan, 0)."""
    b = t_sec // 600
    brk = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    bb = b[brk]
    vol = np.add.reduceat(usdc, brk)
    vw = np.add.reduceat(p0 * usdc, brk) / np.maximum(vol, 1e-9)
    ok = vol >= FINE_BIN_USD
    bb, vw = bb[ok], vw[ok]
    if len(bb) < 3:
        return np.nan, 0.0
    d = np.where(np.diff(bb) <= FINE_GAP_BINS, np.diff(vw), 0.0)
    k = int(np.argmax(np.abs(d)))
    return float(bb[k + 1]), float(d[k])


def process(a):
    data, res = Path(a.data), Path(a.results)
    out = res / "timing"; out.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(a.split, tz="UTC")
    ev = pd.read_parquet(res / "events_table_events.parquet")
    mk = pd.read_parquet(data / "markets_coded.parquet")
    pt = st.period_times(mk); ek = st.event_keys(mk)
    mk["period_time"] = mk["condition_id"].map(ek).map(pt)
    mk = mk[mk["period_time"] >= split].set_index("condition_id")
    n_codes = int(max(ev["wallet"].max(), 1)) + 1
    grp = groups_for(ev, split, n_codes + 10)
    print(f"evaluation markets: {len(mk):,}; wallets: flagged R2 {int(((grp & 2) > 0).sum())}, R0 {int(((grp & 4) > 0).sum())}, "
          f"experienced {int(((grp & 1) > 0).sum()):,}", flush=True)
    pf = pq.ParquetFile(data / "fills_clean.parquet")
    # one row group per market; find which row groups belong to evaluation markets
    first_cond = [pf.metadata.row_group(i).column(1).statistics for i in range(pf.metadata.num_row_groups)]
    rg_of = {}
    for i, s in enumerate(first_cond):
        if s is not None and s.has_min_max and s.min == s.max:
            rg_of[int(s.min)] = i
    conds = [c for c in mk.index if c in rg_of]
    if a.limit:
        conds = conds[:a.limit]
    # online histograms for ALL traders (dollars), per clock
    life_edges = np.linspace(0, 1, 11)
    hres_edges = np.array([0, 1, 6, 24, 72, 168, 720, 2160, 1e9])
    agg = {"life": np.zeros(10), "hres": np.zeros(len(hres_edges) - 1), "rel": np.zeros(len(REL_LABELS)), "rel10": np.zeros(len(FINE_LABELS))}
    t_jump = 0
    parts, rows_kept, t_shift, shifts = [], 0, 0, []
    for n, c in enumerate(conds):
        df = pf.read_row_group(rg_of[c], columns=COLS).to_pandas()
        if df.empty:
            continue
        o = np.argsort(df["timestamp"].to_numpy(), kind="mergesort")
        df = df.iloc[o].reset_index(drop=True)
        t = df["timestamp"].to_numpy("int64"); p0 = df["p0"].to_numpy("float64"); usd = df["usdc"].to_numpy("float64")
        dir0 = df["dir0"].to_numpy("int64"); w = df["wallet"].to_numpy("int64")
        mrow = mk.loc[c]; v0 = float(mrow["value_0"]); res_t = int(pd.Timestamp(mrow["resolve_time"]).timestamp())
        h_star, size = market_shift(t, p0, usd)
        has_shift = np.isfinite(h_star) and abs(size) >= SHIFT_MIN
        rel = np.where(has_shift, (t // 3600) - (h_star if has_shift else 0), np.nan)
        rel_b = np.where(np.isfinite(rel), np.digitize(np.nan_to_num(rel), REL_EDGES), -1)
        b_star, jsize = market_jump(t, p0, usd)
        has_jump = np.isfinite(b_star) and abs(jsize) >= FINE_MIN
        rel10 = np.where(has_jump, ((t // 600) - (b_star if has_jump else 0)) * 10.0, np.nan)      # minutes
        rel10_b = np.where(np.isfinite(rel10), np.digitize(np.nan_to_num(rel10), FINE_EDGES), -1)
        fl = df["frac_life"].to_numpy("float64"); hr_ = df["hours_to_resolve"].to_numpy("float64")
        agg["life"] += np.bincount(np.clip((np.nan_to_num(fl, nan=0.0) * 10).astype(int), 0, 9), weights=usd, minlength=10)
        agg["hres"] += np.bincount(np.clip(np.digitize(np.nan_to_num(hr_, nan=0.0), hres_edges) - 1, 0, len(hres_edges) - 2), weights=usd, minlength=len(hres_edges) - 1)
        if has_shift:
            ok = rel_b >= 0
            agg["rel"] += np.bincount(rel_b[ok], weights=usd[ok], minlength=len(REL_LABELS))
            shifts.append({"condition_id": c, "hour": h_star, "size": size, "volume": usd.sum()})
            t_shift += 1
        if has_jump:
            ok10 = rel10_b >= 0
            agg["rel10"] += np.bincount(rel10_b[ok10], weights=usd[ok10], minlength=len(FINE_LABELS))
            t_jump += 1
        g = grp[w]
        keep = g > 0
        if not keep.any():
            continue
        # held-outcome entry price, payout, markouts (price of the next trade at or after t + horizon)
        long0 = dir0 > 0
        entry = np.where(long0, p0, 1.0 - p0)
        payout = np.where(long0, v0, 1.0 - v0)
        r = {"wallet": w[keep].astype("int32"), "grp": g[keep], "condition_id": np.full(keep.sum(), c, dtype="int32"),
             "ts_sec": t[keep], "usdc": usd[keep].astype("float32"), "entry_price": entry[keep].astype("float32"),
             "edge": (payout - entry)[keep].astype("float32"), "frac_life": fl[keep].astype("float32"),
             "hours_to_resolve": hr_[keep].astype("float32"), "hours_to_info": df["hours_to_info"].to_numpy("float64")[keep].astype("float32"),
             "is_fomc": np.full(keep.sum(), str(mrow.get("info_time_source")) == "fomc_statement"),
             "rel_hours": rel[keep].astype("float32"), "rel_bucket": rel_b[keep].astype("int8"),
             "with_shift": np.where(has_shift, (np.sign(size) * np.sign(dir0)) > 0, False)[keep],
             "shift_size": np.full(keep.sum(), abs(size) if has_shift else np.nan, dtype="float32"),
             "rel10_bucket": rel10_b[keep].astype("int8"),
             "with_jump": np.where(has_jump, (np.sign(jsize) * np.sign(dir0)) > 0, False)[keep],
             "jump_size": np.full(keep.sum(), abs(jsize) if has_jump else np.nan, dtype="float32")}
        for name, h in HORIZONS.items():
            idx = np.searchsorted(t, t[keep] + h, side="left")
            beyond = (idx >= len(t)) | ((t[keep] + h) >= res_t)
            future_p0 = p0[np.minimum(idx, len(t) - 1)]
            fut = np.where(long0[keep], future_p0, 1.0 - future_p0)
            fut = np.where(beyond, payout[keep], fut)
            r[name] = (fut - entry[keep]).astype("float32")
        parts.append(pd.DataFrame(r)); rows_kept += keep.sum()
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(conds)} markets, {rows_kept:,} wallet trades kept", flush=True)
    tr = pd.concat(parts, ignore_index=True)
    tr.to_parquet(out / "timing_trades.parquet", index=False)
    pd.DataFrame(shifts).to_parquet(out / "market_shifts.parquet", index=False)
    json.dump({"split": a.split, "markets": len(conds), "markets_with_shift": t_shift, "markets_with_sharp_jump": t_jump,
               "all_traders_dollars": {k: v.tolist() for k, v in agg.items()}, "hres_edges": hres_edges.tolist(),
               "groups": {k: int(((grp & b) > 0).sum()) for k, b in GROUP_BITS.items()}},
              open(out / "timing_info.json", "w"), indent=1)
    print(f"done: {len(tr):,} wallet trades, {t_shift} of {len(conds)} markets with an hourly price shift of at least {SHIFT_MIN:.2f}, "
          f"{t_jump} with a sharp 10-minute jump of at least {FINE_MIN:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["process", "report"])
    ap.add_argument("--data", default="data_events_full/live")
    ap.add_argument("--results", default="results_v4")
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--limit", type=int, default=0, help="only the first N markets (testing)")
    a = ap.parse_args()
    if a.stage == "process":
        process(a)
    else:
        from make_timing_figures import report
        report(Path(a.results) / "timing")


if __name__ == "__main__":
    main()
