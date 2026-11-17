"""Synthetic Polymarket-like data with KNOWN informed traders.

Purpose: before trusting the pipeline on real data, check that it recovers
planted informed wallets and does NOT flag pure-noise traders. Output is in
the same raw shapes the live APIs return (Gamma market dicts, Data API v2
trade rows), so the whole pipeline is exercised end to end.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _hex(rng, n):
    return "0x" + "".join(rng.choice(list("0123456789abcdef"), n))


def generate(n_markets=160, n_noise=1500, n_informed=50, n_mm=30, seed=473,
             start="2026-01-01", end="2026-09-30", life_days=30):
    rng = np.random.default_rng(seed)
    noise = [_hex(rng, 40) for _ in range(n_noise)]
    informed = [_hex(rng, 40) for _ in range(n_informed)]
    mms = [_hex(rng, 40) for _ in range(n_mm)]
    t_lo, t_hi = pd.Timestamp(start, tz="UTC").value // 10**9, pd.Timestamp(end, tz="UTC").value // 10**9
    life = life_days * 86400

    gamma, rows = [], []
    for _ in range(n_markets):
        cid = _hex(rng, 64)
        tok = [str(int(_hex(rng, 30)[2:], 16)), str(int(_hex(rng, 30)[2:], 16))]
        p_true = rng.uniform(0.15, 0.85)
        y = int(rng.random() < p_true)            # 1 -> outcome 0 ("Yes") wins
        t_res = int(rng.integers(t_lo + life, t_hi))
        t_start = t_res - life
        gamma.append({"conditionId": cid, "id": str(rng.integers(1e6)), "question": "synthetic",
                      "clobTokenIds": json.dumps(tok), "outcomes": json.dumps(["Yes", "No"]),
                      "outcomePrices": json.dumps(["1", "0"] if y else ["0", "1"]),
                      "startDate": pd.Timestamp(t_start, unit="s", tz="UTC").isoformat(),
                      "endDate": pd.Timestamp(t_res, unit="s", tz="UTC").isoformat(),
                      "closedTime": pd.Timestamp(t_res, unit="s", tz="UTC").isoformat(),
                      "volumeNum": 1e5, "negRisk": False, "version": "v1"})

        def price(t):   # public belief converges to the outcome near resolution
            s = ((t - t_start) / life) ** 3
            return np.clip(p_true + (y - p_true) * s + rng.normal(0, 0.03), 0.02, 0.98)

        def emit(wallet, t, dir0, usdc, spread):
            p0 = float(np.clip(price(t) + dir0 * spread, 0.01, 0.99))
            if rng.random() < 0.5:   # express as outcome-0 token
                o, side, px = 0, "BUY" if dir0 > 0 else "SELL", p0
            else:                    # or as the complementary outcome-1 token
                o, side, px = 1, "SELL" if dir0 > 0 else "BUY", 1 - p0
            rows.append({"proxy_wallet": wallet, "condition_id": cid, "token_id": tok[o],
                         "side": side, "price": round(px, 4), "size": round(usdc / px, 2),
                         "timestamp": int(t), "transaction_hash": _hex(rng, 64), "outcome_index": o})

        for w in rng.choice(noise, rng.integers(40, 120), replace=False):
            for _ in range(rng.integers(1, 4)):
                emit(w, rng.integers(t_start, t_res), rng.choice([-1, 1]),
                     rng.lognormal(3.5, 1.0), 0.01)
        for w in mms:
            if rng.random() < 0.6:
                for _ in range(rng.integers(4, 12)):
                    emit(w, rng.integers(t_start, t_res), rng.choice([-1, 1]),
                         rng.lognormal(4.5, 0.6), -0.008)   # earns half-spread
        for w in informed:
            if rng.random() < 0.35:
                signal = (1 if y else -1) * (1 if rng.random() < 0.8 else -1)   # 80% accurate
                t0 = t_res - int(life * rng.uniform(0.15, 0.5))   # before the news is priced in
                for k in range(rng.integers(3, 8)):          # split into child orders
                    emit(w, min(t0 + k * int(rng.integers(30, 240)), t_res - 1), signal,
                         rng.lognormal(5.0, 0.5), 0.012)

    truth = pd.Series("noise", index=noise)
    truth = pd.concat([truth, pd.Series("informed", index=informed), pd.Series("mm", index=mms)])
    return gamma, pd.DataFrame(rows), truth
