import pandas as pd
import pytest

from pm473 import onchain, pnl


def _mk(v0, v1):
    return pd.DataFrame([{"condition_id": "c", "value_0": v0, "value_1": v1,
                          "resolve_time": pd.Timestamp("2026-06-01", tz="UTC")}])


def _fills(rows):
    df = pd.DataFrame(rows, columns=["wallet", "outcome_index", "side", "size", "price"])
    df["condition_id"] = "c"
    df["usdc"] = df["size"] * df["price"]
    df["fee_usdc"] = 0.0
    return df


def test_simple_buy_win():
    r = pnl.wallet_market_pnl(_fills([("a", 0, "BUY", 100, 0.4)]), _mk(1, 0)).iloc[0]
    assert r.pnl == pytest.approx(60)          # paid 40, redeemed 100
    assert r.roi == pytest.approx(1.5)


def test_sell_without_buy_is_split():
    # split $1 -> YES+NO, sell YES at 0.6, keep NO. NO wins -> +0.6 ; YES wins -> -0.4
    f = _fills([("a", 0, "SELL", 1, 0.6)])
    assert pnl.wallet_market_pnl(f, _mk(0, 1)).iloc[0].pnl == pytest.approx(0.6)
    assert pnl.wallet_market_pnl(f, _mk(1, 0)).iloc[0].pnl == pytest.approx(-0.4)


def test_round_trip_and_merge():
    # buy YES .3, buy NO .6 (=merge-able pair for .9) -> +0.1 regardless of outcome
    f = _fills([("a", 0, "BUY", 10, 0.3), ("a", 1, "BUY", 10, 0.6)])
    for m in (_mk(1, 0), _mk(0, 1)):
        assert pnl.wallet_market_pnl(f, m).iloc[0].pnl == pytest.approx(1.0)


def test_zero_sum_between_counterparties():
    f = _fills([("a", 0, "BUY", 50, 0.7), ("b", 0, "SELL", 50, 0.7)])
    out = pnl.wallet_market_pnl(f, _mk(0, 1))
    assert out.pnl.sum() == pytest.approx(0)


def _word(x):
    return f"{x:064x}"


def test_decode_v2_buy():
    tok = 123456789
    log = {"address": "0xe111180000d2663c0091e4f400237545b87b996b", "transactionHash": "0xab",
           "logIndex": "0x3",
           "topics": [onchain.T_V2, "0x" + "11" * 32, "0x" + "0" * 24 + "aa" * 20, "0x" + "0" * 24 + "bb" * 20],
           "data": "0x" + _word(0) + _word(tok) + _word(40_000_000) + _word(100_000_000)
                   + _word(0) + _word(0) + _word(0)}
    d = onchain.decode_order_filled(log)
    assert d["side"] == "BUY" and d["token_id"] == str(tok)
    assert d["shares"] == 100 and d["usdc"] == 40 and d["price"] == pytest.approx(0.4)
    assert d["maker"] == "0x" + "aa" * 20


def test_decode_v1_sell():
    tok = 987
    log = {"address": "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e", "transactionHash": "0xcd",
           "logIndex": "0x0",
           "topics": [onchain.T_V1, "0x" + "11" * 32, "0x" + "0" * 24 + "aa" * 20, "0x" + "0" * 24 + "bb" * 20],
           "data": "0x" + _word(tok) + _word(0) + _word(50_000_000) + _word(30_000_000) + _word(0)}
    d = onchain.decode_order_filled(log)
    assert d["side"] == "SELL" and d["price"] == pytest.approx(0.6)


def test_non_orderfilled_exchange_logs_are_skipped():
    for topics in (["0x" + "ab" * 32], ["0x" + "ab" * 32, "0x" + "11" * 32],
                   [onchain.T_V2, "0x" + "11" * 32]):          # too few topics
        log = {"address": "0xe111180000d2663c0091e4f400237545b87b996b", "transactionHash": "0x1",
               "logIndex": "0x0", "topics": topics, "data": "0x"}
        assert onchain.decode_order_filled(log) is None


def test_gamma_paging_survives_server_page_cap():
    from pm473.api import Client
    allm = [{"id": str(i)} for i in range(730)]

    class Fake(Client):
        def get(self, base, path, params=None):
            from pm473.api import APIError
            if path == "/markets/keyset":
                raise APIError("404 not found")              # forces the legacy path
            lim = min(params["limit"], 100)                  # server caps pages at 100
            return allm[params["offset"]:params["offset"] + lim]

    got = list(Fake().gamma_markets(page_size=500))
    assert len(got) == 730


def _gm(i, day, vol=1e5):
    return {"id": str(i), "closed": True, "volumeNum": vol, "endDate": f"2026-{day}T00:00:00Z"}


def test_keyset_uses_after_cursor():
    from pm473.api import Client
    allm = [_gm(i, "03-01") for i in range(250)]

    class Fake(Client):
        def get(self, base, path, params=None):
            assert path == "/markets/keyset" and "cursor" not in params
            start = int(params.get("after_cursor", 0))
            page = allm[start:start + params["limit"]]
            nxt = str(start + len(page)) if start + len(page) < len(allm) else None
            return {"markets": page, "next_cursor": nxt}

    got = list(Fake().gamma_markets(closed=True, volume_num_min=5e4,
                                    end_date_min="2026-01-01", end_date_max="2026-09-30"))
    assert len(got) == 250


def test_falls_back_to_date_windows_when_keyset_replays():
    from pm473.api import Client
    days = [f"{m:02d}-{d:02d}" for m in range(1, 10) for d in (5, 15, 25)]
    allm = [_gm(i, days[i % len(days)]) for i in range(5400)]      # > 2000 -> needs splitting

    class Fake(Client):
        def get(self, base, path, params=None):
            if path == "/markets/keyset":                          # broken: same page forever
                return {"markets": allm[:100], "next_cursor": "same"}
            if params["offset"] >= 2000:
                raise APIError("422 offset too large, use /markets/keyset")
            lo, hi = params["end_date_min"], params["end_date_max"]
            sel = [m for m in allm if lo <= m["endDate"] <= hi]
            return sel[params["offset"]:params["offset"] + params["limit"]]

    from pm473.api import APIError
    got = list(Fake().gamma_markets(closed=True, end_date_min="2026-01-01", end_date_max="2026-09-30"))
    assert len({m["id"] for m in got}) == 5400


def test_skill_counts_events_not_outcome_markets():
    from pm473 import classify
    rows = []
    # wallet a: 4 outcome markets of ONE election, plus 2 standalone markets
    for i in range(4):
        rows.append({"wallet": "a", "condition_id": f"e1-{i}", "event_key": "e1", "pnl": 10.0, "capital": 100.0})
    for j in range(2):
        rows.append({"wallet": "a", "condition_id": f"s{j}", "event_key": f"s{j}", "pnl": 5.0 + j, "capital": 100.0})
    s = classify.wallet_skill(pd.DataFrame(rows), min_markets=1)
    assert s.loc["a", "skill_n"] == 3                     # 1 event + 2 standalone
    assert s.loc["a", "n_outcome_markets"] == 6
    assert s.loc["a", "skill_mean_roi"] == pytest.approx((0.10 + 0.05 + 0.06) / 3)   # event ROI = 40/400


def test_identical_fills_are_kept():
    import json
    from pm473 import clean, collect
    g = {"id": "1", "conditionId": "0x" + "c" * 64, "question": "q", "slug": "s",
         "clobTokenIds": json.dumps(["11", "12"]), "outcomes": json.dumps(["Yes", "No"]),
         "outcomePrices": json.dumps(["1", "0"]), "startDate": "2026-05-01T00:00:00Z",
         "endDate": "2026-06-01T00:00:00Z", "closedTime": "2026-06-01T00:00:00Z", "volumeNum": 1e5}
    m, _ = collect.filter_markets([g])
    row = {"proxy_wallet": "0x" + "a" * 40, "condition_id": "0x" + "c" * 64, "token_id": "11",
           "side": "BUY", "price": 0.5, "size": 10, "timestamp": 1779000000, "transaction_hash": "0xt"}
    f, log = clean.normalize(pd.DataFrame([row, row, row]), m)
    assert len(f) == 3 and log["identical_rows_kept"] == 2
