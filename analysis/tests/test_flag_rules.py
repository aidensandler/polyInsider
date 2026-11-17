import numpy as np
import pandas as pd

from pm473 import skill_test as st


def _stats(seed=0, n=3000, spread=0.6, centre=0.2, n_skilled=15):
    rng = np.random.default_rng(seed)
    z = rng.normal(centre, spread, n)
    z[:n_skilled] += 4.0
    return pd.DataFrame({"Z": z, "n_events": 20, "eligible": True})


def test_default_rule_is_unchanged():
    s = st.flag_wallets(_stats())
    assert s.attrs["sigma"] == 1.0 and s.attrs["center"] == 0.0


def test_narrow_null_flags_more_and_stays_clean_without_skill():
    a = st.flag_wallets(_stats())
    b = st.flag_wallets(_stats(), sigma_floor=0.0, center=True)
    assert b.attrs["sigma"] < 0.8 and abs(b.attrs["center"] - 0.2) < 0.1
    assert b["informed"].sum() >= a["informed"].sum()
    clean = st.flag_wallets(_stats(n_skilled=0), sigma_floor=0.0, center=True)
    assert clean["informed"].sum() <= 2             # pure noise: essentially nothing flagged
