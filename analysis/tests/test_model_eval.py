"""Tests for model_eval.py (uses the synthetic generator; no network)."""
import numpy as np

from pm473 import model_eval as me, skill_test as st
from test_skill_test import _synthetic


def _events(n_informed):
    fills, mk, names, truth = _synthetic(n_informed=n_informed, n_markets=320)
    return st.event_table(st.wallet_market_terms(fills, mk), mk)


def test_planted_traders_persist_out_of_sample():
    ev = _events(60)
    split = ev["period_time"].quantile(0.5)
    m, panel, dec = me.performance(ev, split, min_events=5, min_eval_events=2)
    assert m["spearman_Z_vs_eval_mean_z"]["rho"] > 0.05
    assert m["auc_beat_prices"]["Z"] > 0.52
    assert len(dec) == 10 and dec["mean_eval_mean_z"].iloc[-1] > dec["mean_eval_mean_z"].iloc[0]


def test_noise_world_has_no_persistence():
    ev = _events(0)
    split = ev["period_time"].quantile(0.5)
    m, _, _ = me.performance(ev, split, min_events=5, min_eval_events=2)
    assert abs(m["spearman_Z_vs_eval_mean_z"]["rho"]) < 0.12
    assert abs(m["auc_beat_prices"]["Z"] - 0.5) < 0.06


def test_walk_forward_runs():
    ev = _events(60)
    lo, hi = ev["period_time"].min(), ev["period_time"].max()
    starts = [str((lo + (hi - lo) * f).date()) for f in (0.4, 0.6)]
    wf = me.walk_forward(ev, starts, window_days=60, min_events=5, min_eval_events=2)
    assert len(wf) == 2 and wf["split"].tolist() == starts
