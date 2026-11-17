"""Offline tests for live_feed.py (no network)."""
import json

import pytest

import live_feed as lf
from pm473.api import APIError


ROW = {"proxy_wallet": "0xAbC", "condition_id": "0xC1", "token_id": "111", "side": "buy",
       "price": "0.25", "size": "40", "timestamp": 1_780_000_000, "transaction_hash": "0xtx"}


def test_normalize_and_reject_bad_rows():
    t = lf.normalize_row(ROW)
    assert t["wallet"] == "0xabc" and t["condition_id"] == "0xc1" and t["side"] == "BUY"
    assert lf.normalize_row({**ROW, "price": 1.2}) is None
    assert lf.normalize_row({**ROW, "side": "HOLD"}) is None
    ms = lf.normalize_row({**ROW, "timestamp": 1_780_000_000_000})
    assert ms["ts"] == 1_780_000_000


def test_held_return():
    r, p = lf.held_return("BUY", 0.25, 40, 0, 0)          # bought winner at 0.25
    assert r == pytest.approx(3.0) and p == pytest.approx(30.0)
    r, p = lf.held_return("SELL", 0.25, 40, 0, 1)         # sold outcome 0, outcome 1 won -> long No at 0.75
    assert r == pytest.approx(1 / 3) and p == pytest.approx(10.0)
    r, p = lf.held_return("BUY", 0.25, 40, 0, 1)          # bought loser
    assert r == pytest.approx(-1.0) and p == pytest.approx(-10.0)


def test_add_new_dedupes_and_filters():
    st = {"seen": [], "trades": [], "markets": {}, "param": None}
    rows = [ROW, ROW, {**ROW, "proxy_wallet": "0xother"}, {**ROW, "timestamp": 1_000}]
    assert lf.add_new(st, rows, {"0xabc"}, since_ts=1_700_000_000) == 1
    assert lf.add_new(st, rows, {"0xabc"}, since_ts=1_700_000_000) == 0
    assert st["trades"][0]["status"] == "open"


class FakeClient:
    def __init__(self, behaviour):
        self.b = behaviour

    def paginate_v2(self, path, params, max_pages=None):
        key = [k for k in params if k not in ("limit", "taker_only")][0]
        res = self.b[key]
        if isinstance(res, Exception):
            raise res
        return iter(res)


def test_probe_picks_the_param_that_filters():
    w = "0xabc"
    c = FakeClient({"user": APIError("400 unknown param"),
                    "proxy_wallet": [{"proxy_wallet": "0xzzz"}, {"proxy_wallet": w}],   # ignores filter
                    "wallet": [{"proxy_wallet": w}, {"proxy_wallet": w}]})
    assert lf.probe(c, w, ["user", "proxy_wallet", "wallet"]) == "wallet"
    assert lf.probe(FakeClient({"user": APIError("x")}), w, ["user"]) is None


def test_settle_resolves_and_exports(tmp_path):
    st = {"seen": [], "trades": [], "markets": {}, "param": None}
    lf.add_new(st, [ROW], {"0xabc"}, 0)
    st["markets"]["0xc1"] = {"token_0": "111", "token_1": "222", "winner_index": 0, "question": "Will X?",
                             "event_slug": "x", "outcome_0": "Yes", "outcome_1": "No", "closed": True,
                             "closed_checked": 9_999_999_999}
    lf.settle(None, st)
    t = st["trades"][0]
    assert t["status"] == "resolved" and t["return"] == pytest.approx(3.0)
    lf.export(st, tmp_path / "live.json")
    out = json.loads((tmp_path / "live.json").read_text())["trades"][0]
    assert out["market"] == "Will X?" and out["held"] == "Yes" and out["status"] == "resolved"
