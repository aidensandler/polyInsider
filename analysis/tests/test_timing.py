import numpy as np

import timing_analysis as ta


def _tape(jump_at_s=30 * 3600, n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 60 * 3600, n)).astype("int64")
    p = np.where(t < jump_at_s, 0.30, 0.80) + rng.normal(0, 0.005, n)
    return t, p.astype("float64"), np.full(n, 500.0)


def test_hourly_shift_found_at_the_jump_hour():
    t, p, u = _tape()
    h, size = ta.market_shift(t, p, u)
    assert abs(h - 30) <= 1 and 0.4 < size < 0.6


def test_ten_minute_jump_found_and_signed():
    t, p, u = _tape(jump_at_s=20 * 3600 + 1234)
    b, size = ta.market_jump(t, p, u)
    assert abs(b * 600 - (20 * 3600 + 1234)) <= 1800 and size > 0.4
    b2, size2 = ta.market_jump(t, 1.0 - p, u)           # price falling instead
    assert size2 < -0.4


def test_no_shift_in_a_flat_market():
    t = np.arange(0, 50 * 3600, 300, dtype="int64")
    p = np.full(len(t), 0.5)
    h, size = ta.market_shift(t, p, np.full(len(t), 500.0))
    assert abs(size) < 0.01


def test_group_flags_use_only_formation_events():
    import pandas as pd
    rng = np.random.default_rng(1)
    rows = []
    for w in range(300):
        for e in range(12):
            z = rng.normal(0.2, 0.6) + (4.0 if w < 8 else 0.0)
            rows.append({"wallet": w, "event_key": f"e{e}", "period_time": pd.Timestamp("2026-03-01", tz="UTC") + pd.Timedelta(days=e),
                         "c": z, "V": 1.0, "volume": 100.0, "n_fills": 3, "z": z, "z_clipped": float(np.clip(z, -3, 3))})
    ev = pd.DataFrame(rows)
    g = ta.groups_for(ev, pd.Timestamp("2026-06-01", tz="UTC"), 400)
    assert (g[:300] & 1).all() and (g[:8] & 2).all() and (g[8:300] & 2).sum() <= 2
