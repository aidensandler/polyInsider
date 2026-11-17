import numpy as np
import pandas as pd

from pm473 import alerts as al


def _table(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp("2026-03-01", tz="UTC") + pd.to_timedelta(np.sort(rng.integers(0, 40 * 86400, n)), unit="s")
    c = pd.DataFrame({
        "wallet": rng.integers(0, 40, n).astype("int32"),
        "condition_id": rng.integers(0, 12, n).astype("int32"),
        "ts": ts,
        "usdc": rng.lognormal(4, 1, n).astype("float32"),
        "held_outcome": rng.integers(0, 2, n).astype("int8"),
        "entry_price": rng.uniform(0.05, 0.95, n).astype("float32"),
        "log_hours_to_sched": rng.uniform(0, 7, n).astype("float32"),
        "Z_past": rng.normal(1, 2, n).astype("float32"),
        "n_past": rng.integers(3, 30, n).astype("float32"),
        "edge": rng.normal(0, 0.4, n).astype("float32"),
    })
    c["log_usdc"] = np.log1p(c["usdc"]).astype("float32")
    return c


def test_context_features_use_only_the_past():
    c = _table()
    full = al.add_context_features(c)
    cut = c["ts"].iloc[2500]
    part = al.add_context_features(c[c["ts"] < cut])
    assert len(part) > 100
    a = full[full["ts"] < cut].reset_index(drop=True)
    b = part.reset_index(drop=True)
    for col in al.CONTEXT_FEATURES + al.FLOW_FEATURES:
        np.testing.assert_allclose(a[col].to_numpy("float64"), b[col].to_numpy("float64"), rtol=1e-4, atol=1e-5,
                                   equal_nan=True, err_msg=col)


def test_labels_do_not_change_features():
    c = _table()
    a = al.add_context_features(c)
    c2 = c.copy()
    c2["edge"] = -c2["edge"]
    b = al.add_context_features(c2)
    for col in al.CONTEXT_FEATURES + al.FLOW_FEATURES:
        np.testing.assert_array_equal(a[col].to_numpy(), b[col].to_numpy())


def test_agreement_counts_others_only():
    t0 = pd.Timestamp("2026-03-01", tz="UTC")
    rows = [  # wallet, condition, seconds, held, usdc
        (1, 0, 0, 0, 100.0), (2, 0, 600, 0, 50.0), (1, 0, 900, 0, 70.0), (3, 0, 1000, 1, 40.0), (4, 0, 1200, 0, 10.0)]
    c = pd.DataFrame({"wallet": [r[0] for r in rows], "condition_id": [r[1] for r in rows],
                      "ts": [t0 + pd.Timedelta(seconds=r[2]) for r in rows], "held_outcome": [r[3] for r in rows],
                      "usdc": [r[4] for r in rows]})
    c["held_outcome"] = c["held_outcome"].astype("int8")
    c["log_usdc"] = np.log1p(c["usdc"])
    c["entry_price"], c["log_hours_to_sched"], c["Z_past"], c["n_past"] = 0.5, 3.0, 5.0, 10.0   # everyone is 'good'
    r = al.add_context_features(c)
    # last trade (wallet 4, outcome 0): earlier good dollars on outcome 0 by others = 100 + 50 + 70; on outcome 1 = 40
    last = r.iloc[-1]
    assert np.isclose(np.expm1(last["agree_log_usd_all"]), 220.0, rtol=1e-4)
    assert np.isclose(np.expm1(last["oppose_log_usd_24h"]), 40.0, rtol=1e-4)
    assert last["agree_n_1h"] == 3
    assert np.isclose(np.expm1(last["flow_log_usd_all"]), 220.0, rtol=1e-4)     # everyone is good here, so flow = agreement
    assert np.isclose(np.expm1(last["flow_opp_log_usd_24h"]), 40.0, rtol=1e-4)
    # third trade (wallet 1 again): its own first trade is excluded, only wallet 2's 50 counts
    third = r.iloc[2]
    assert np.isclose(np.expm1(third["agree_log_usd_all"]), 50.0, rtol=1e-4)
    assert third["pos_n_same"] == 1
    # first trade has no history
    assert r.iloc[0]["agree_log_usd_all"] == 0 and np.isnan(r.iloc[0]["lw_two_sided_past"])


def test_two_sided_flag():
    t0 = pd.Timestamp("2026-03-01", tz="UTC")
    c = pd.DataFrame({"wallet": [1, 1, 1], "condition_id": [0, 0, 0],
                      "ts": [t0 + pd.Timedelta(seconds=s) for s in (0, 60, 120)],
                      "held_outcome": np.array([0, 1, 0], dtype="int8"), "usdc": [10.0, 10.0, 10.0]})
    c["log_usdc"] = np.log1p(c["usdc"])
    c["entry_price"], c["log_hours_to_sched"], c["Z_past"], c["n_past"] = 0.5, 3.0, 0.0, 3.0
    r = al.add_context_features(c)
    assert list(r["opp_in_mkt"]) == [0, 1, 1]
    assert np.isclose(r.iloc[2]["lw_two_sided_past"], 0.5)      # one of the two earlier trades was two-sided
