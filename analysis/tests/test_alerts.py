"""Alert model: no look-ahead in features, clean train/test split, and correct copy-trade arithmetic."""
import numpy as np
import pandas as pd
import pytest

from pm473 import alert_backtest as bt
from pm473 import alerts as al

T = lambda s: pd.Timestamp(s, tz="UTC")


def make_events():
    # wallet 1 resolves events on Jan 10, Jan 20, Jan 30; wallet 2 only on Jan 10
    rows = [(1, T("2026-01-10"), 5.0, 100.0, 2.0), (1, T("2026-01-20"), -1.0, 50.0, -0.5), (1, T("2026-01-30"), 2.0, 40.0, 1.0),
            (2, T("2026-01-10"), 1.0, 10.0, 0.5)]
    return pd.DataFrame(rows, columns=["wallet", "period_time", "c", "volume", "z_clipped"])


def test_history_uses_only_events_resolved_strictly_before_the_trade():
    hist = al.wallet_history(make_events())
    trades = pd.DataFrame({"wallet": [1, 1, 1, 1, 2, 3],
                           "ts": [T("2026-01-05"), T("2026-01-10"), T("2026-01-15"), T("2026-02-01"), T("2026-01-15"), T("2026-01-15")]})
    out = al.add_wallet_features(trades, hist).sort_values(["wallet", "ts"]).reset_index(drop=True)
    assert out["n_past"].tolist() == [0, 0, 1, 3, 1, 0]            # an event resolved AT the trade time is not yet known
    assert out.loc[2, "Z_past"] == pytest.approx(2.0 / np.sqrt(1))
    assert out.loc[3, "Z_past"] == pytest.approx((2.0 - 0.5 + 1.0) / np.sqrt(3))
    assert out.loc[3, "win_rate_past"] == pytest.approx(2 / 3)
    assert out.loc[3, "ppd_past"] == pytest.approx((5.0 - 1.0 + 2.0) / (100 + 50 + 40))
    assert np.isnan(out.loc[0, "Z_past"]) and np.isnan(out.loc[5, "Z_past"])


def test_future_events_never_change_earlier_features():
    ev = make_events()
    trades = pd.DataFrame({"wallet": [1], "ts": [T("2026-01-25")]})
    a = al.add_wallet_features(trades, al.wallet_history(ev))
    ev2 = pd.concat([ev, pd.DataFrame([(1, T("2026-03-01"), 99.0, 1.0, 3.0)], columns=ev.columns)])
    b = al.add_wallet_features(trades, al.wallet_history(ev2))
    assert a[["n_past", "Z_past", "ppd_past"]].equals(b[["n_past", "Z_past", "ppd_past"]])


def test_train_test_split_has_no_overlap_or_lookahead():
    c = pd.DataFrame({"ts": [T("2026-05-01"), T("2026-05-20"), T("2026-05-25"), T("2026-06-02"), T("2026-06-10")],
                      "event_time": [T("2026-05-10"), T("2026-05-28"), T("2026-07-01"), T("2026-06-05"), T("2026-07-01")]})
    tr, te = al.split_train_test(c, "2026-06-01")
    assert tr.index.tolist() == [0, 1]                    # made before the cut-off AND resolved by it
    assert te.index.tolist() == [3, 4]
    assert (tr["ts"] < T("2026-06-01")).all() and (tr["event_time"] < T("2026-06-01")).all() and (te["ts"] >= T("2026-06-01")).all()


def test_scheduled_time_ignores_early_resolution():
    mk = pd.DataFrame({"condition_id": [0, 1, 2], "end_time": [T("2026-03-01"), T("2026-03-01"), T("2026-03-01")],
                       "info_time": [T("2026-02-10"), T("2026-03-01"), T("2026-03-05")],
                       "info_time_source": ["scheduled_end", "scheduled_end", "fomc_statement"]})
    s = al.scheduled_times(mk)
    assert s[0] == T("2026-03-01")            # info_time was capped at an early resolve; the scheduled end is used
    assert s[2] == T("2026-03-05")            # FOMC / event markets keep their scheduled time


# ----------------------------------------------------------------------------- backtest
def make_alerts():
    return pd.DataFrame({"wallet": [1, 1, 1, 2], "condition_id": [10, 10, 10, 10], "held_outcome": [0, 0, 1, 0],
                         "ts": [T("2026-06-01 10:00"), T("2026-06-01 10:30"), T("2026-06-01 10:40"), T("2026-06-01 11:00")],
                         "entry_price": [0.40, 0.45, 0.60, 0.42], "payout": [1.0, 1.0, 0.0, 1.0], "event_key": ["e:1"] * 4})


def test_dedupe_keeps_first_trade_per_wallet_market_side():
    d = bt.dedupe_alerts(make_alerts())
    assert len(d) == 3 and d.loc[d.wallet == 1, "entry_price"].tolist() == [0.40, 0.60]


def test_follower_pays_next_trade_after_delay_in_held_outcome_terms():
    a = bt.dedupe_alerts(make_alerts())
    tape = pd.DataFrame({"condition_id": [10, 10, 10], "ts": [T("2026-06-01 10:02"), T("2026-06-01 10:06"), T("2026-06-01 10:50")], "p0": [0.41, 0.47, 0.48]})
    tape = bt.make_tape(tape, [10], "2026-06-01")
    p = bt.exec_price(a.reset_index(drop=True), tape, 300)        # 5 minutes after each alert
    # alert 1 (10:00, outcome 0): next trade at or after 10:05 is 10:06 at p0 = 0.47
    # alert 2 (10:40, outcome 1): next trade at or after 10:45 is 10:50 at p0 = 0.48 -> held outcome 1 costs 1 - 0.48
    # alert 3 (11:00, outcome 0): no later trade -> NaN (cannot copy)
    assert p[0] == pytest.approx(0.47) and p[1] == pytest.approx(0.52) and np.isnan(p[2])
    assert (bt.exec_price(a.reset_index(drop=True), tape, 0) == a["entry_price"].to_numpy()).all()


def test_return_and_summary():
    a = bt.dedupe_alerts(make_alerts()).reset_index(drop=True)
    tape = bt.make_tape(pd.DataFrame({"condition_id": [10, 10], "ts": [T("2026-06-01 10:06"), T("2026-06-01 10:50")], "p0": [0.50, 0.50]}), [10], "2026-06-01")
    r = bt.follower_results(a, tape, 300, stake=100.0, slippage=0.0)
    assert len(r) == 2                                                  # the 11:00 alert has no later trade
    first = r[r["held_outcome"] == 0].iloc[0]
    assert first["ret"] == pytest.approx((1.0 - 0.5) / 0.5) and first["pnl"] == pytest.approx(100.0)
    second = r[r["held_outcome"] == 1].iloc[0]                         # held outcome 1 paid 0.50 and lost
    assert second["ret"] == pytest.approx(-1.0) and second["pnl"] == pytest.approx(-100.0)
    s = bt.summarize(r, n_boot=50)
    assert s["alerts"] == 2 and s["roi"] == pytest.approx(0.0) and s["stake_total"] == 200.0
    r2 = bt.follower_results(a, tape, 300, stake=100.0, slippage=0.02)
    assert r2.loc[r2.held_outcome == 0, "ret"].iloc[0] == pytest.approx((1.0 - 0.52) / 0.52)


def test_prices_outside_the_tradable_range_are_skipped():
    a = make_alerts().iloc[:1].copy(); a["entry_price"] = 0.995
    assert bt.follower_results(a, bt.make_tape(pd.DataFrame({"condition_id": [10], "ts": [T("2026-06-01 12:00")], "p0": [0.5]}), [10], "2026-06-01"), 0).empty


def test_model_trains_and_scores_without_touching_test_rows():
    rng = np.random.default_rng(0)
    n = 6000
    d = pd.DataFrame({f: rng.normal(size=n) for f in al.FEATURES})
    d["ts"] = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    d["edge"] = 0.3 * d["Z_past"] + rng.normal(scale=0.2, size=n)
    m, info = al.train_alert_model(d, iter_grid=(20, 40))
    s = al.score(m, d)
    assert np.corrcoef(s, d["edge"])[0, 1] > 0.5 and np.isfinite(info["threshold"]) and info["train_rows"] == n
