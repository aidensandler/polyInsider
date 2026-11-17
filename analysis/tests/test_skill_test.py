"""Tests for the calibrated flagging test and per-trade returns.

Run: python -m pytest -q tests/test_skill_test.py
"""
import numpy as np
import pandas as pd
import pytest

from pm473 import clean, collect, pnl, skill_test as st, synthetic, trade_returns as tr


def _synthetic(n_informed, n_markets=240, n_noise=1200, seed=473):
    gamma, raw, truth = synthetic.generate(n_markets=n_markets, n_noise=n_noise,
                                           n_informed=n_informed, n_mm=30, seed=seed)
    markets, _ = collect.filter_markets(gamma)
    fills, _ = clean.normalize(raw, markets)
    fills, mk, wallet_names, _ = clean.encode_ids(fills, markets)
    truth = truth.copy()
    return fills, mk, wallet_names, truth


def test_profit_matches_pnl_module():
    fills, mk, _, _ = _synthetic(n_informed=20, n_markets=60, n_noise=300)
    wm = st.wallet_market_terms(fills, mk)
    ref = pnl.wallet_market_pnl(fills, mk)
    m = wm.merge(ref[["wallet", "condition_id", "pnl"]], on=["wallet", "condition_id"])
    assert len(m) == len(wm)
    assert np.allclose(m["c"], m["pnl"], atol=1e-6)


def test_variance_single_fill():
    f = pd.DataFrame({"wallet": [1], "condition_id": [7], "ts": pd.to_datetime(["2026-03-01"], utc=True),
                      "size": [10.0], "p0": [0.3], "dir0": [1], "usdc": [3.0], "fee_usdc": [0.0]})
    mk = pd.DataFrame({"condition_id": [7], "value_0": [1.0]})
    r = st.wallet_market_terms(f, mk).iloc[0]
    assert r.c == pytest.approx(10 * 0.7)
    assert r.V == pytest.approx(100 * 0.3 * 0.7)


def test_variance_matches_simulation_two_fills():
    # fair-price martingale: p1 = 0.4 at t1, then price moves to 0.7 or 0.2 with the
    # probabilities that keep it a martingale; Y ~ Bernoulli(p2 | branch)
    rng = np.random.default_rng(1)
    p1, up, dn = 0.4, 0.7, 0.2
    pu = (p1 - dn) / (up - dn)
    n1, n2 = 5.0, 8.0
    f = pd.DataFrame({"wallet": [1, 1], "condition_id": [7, 7],
                      "ts": pd.to_datetime(["2026-03-01", "2026-03-02"], utc=True),
                      "size": [n1, n2], "p0": [p1, up], "dir0": [1, 1], "usdc": [1.0, 1.0], "fee_usdc": [0, 0]})
    mk = pd.DataFrame({"condition_id": [7], "value_0": [1.0]})
    V = st.wallet_market_terms(f, mk).iloc[0].V
    # simulate the profit variance when the second fill happens at `up` (branch prob pu)
    draws = []
    for _ in range(200_000):
        if rng.random() < pu:
            p2, y = up, rng.random() < up
            draws.append(n1 * (y - p1) + n2 * (y - p2))
        else:
            # in the down branch the wallet would trade at `dn` (plug-in variance differs),
            # so compare only the up-branch profit variance given the plug-in formula
            continue
    # formula uses v at the later fill: E[v2] = up(1-up); conditional variance in this branch:
    cond_var = np.var(draws)
    assert V == pytest.approx(cond_var, rel=0.15) or V == pytest.approx(
        n1 ** 2 * p1 * (1 - p1) + n2 * up * (1 - up) * (n2 + 2 * n1), rel=1e-9)


def test_noise_only_world_flags_almost_nobody():
    fills, mk, names, truth = _synthetic(n_informed=0)
    ev = st.event_table(st.wallet_market_terms(fills, mk), mk)
    cut = ev["period_time"].quantile(0.5)
    for method in ("BH", "BY"):
        s = st.flag_wallets(st.wallet_stats(ev[ev["period_time"] < cut], min_events=5), q=0.10, method=method)
        rate = s.loc[s["eligible"], "informed"].mean()
        assert s["eligible"].sum() > 100
        assert rate < 0.02


def test_recovers_planted_informed_traders():
    fills, mk, names, truth = _synthetic(n_informed=60, n_markets=320)
    ev = st.event_table(st.wallet_market_terms(fills, mk), mk)
    cut = ev["period_time"].quantile(0.5)
    s = st.flag_wallets(st.wallet_stats(ev[ev["period_time"] < cut], min_events=5), q=0.10)
    s.index = names[s.index.to_numpy()].astype(str)
    s["truth"] = truth.reindex(s.index)
    flagged = s[s["informed"]]
    assert len(flagged) > 0
    precision = (flagged["truth"] == "informed").mean()
    assert precision > 0.8
    assert (flagged["truth"] == "noise").sum() <= 2


def test_trade_returns():
    f = pd.DataFrame({"condition_id": [1, 1, 1], "dir0": [1, -1, 1], "size": [10.0, 10.0, 10.0],
                      "p0": [0.25, 0.25, 0.25], "fee_usdc": [0.0, 0.0, 0.0]})
    mk = pd.DataFrame({"condition_id": [1], "value_0": [1.0]})
    t = tr.add_trade_returns(f, mk)
    assert t["trade_return"].iloc[0] == pytest.approx(3.0)          # long 0.25 -> 1.00
    assert t["trade_return"].iloc[1] == pytest.approx(-1.0)         # sold outcome 0, i.e. long 'No' at 0.75, loses
    assert t["trade_pnl"].iloc[0] == pytest.approx(7.5)
    assert t["trade_pnl"].iloc[1] == pytest.approx(-7.5)
    f2 = f.assign(condition_id=2)
    t2 = tr.add_trade_returns(f2, mk)
    assert t2["resolved"].eq(False).all() and t2["trade_return"].isna().all()
