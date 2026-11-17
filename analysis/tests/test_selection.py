import json

import pandas as pd

from pm473 import collect
from pm473.api import Client


def _g(i, q, slug, start, end, neg=False, other=False, ev_start=None, vol=1e5):
    return {"id": str(i), "conditionId": f"0x{i:064x}", "question": q, "slug": slug,
            "clobTokenIds": json.dumps([str(10 * i), str(10 * i + 1)]),
            "outcomes": json.dumps(["Yes", "No"]), "outcomePrices": json.dumps(["1", "0"]),
            "startDate": start, "endDate": end, "closedTime": end, "volumeNum": vol,
            "negRisk": neg, "negRiskOther": other, "version": "v1", "closed": True,
            "eventStartTime": ev_start}


RAW = [
    _g(1, "Fed decreases interest rates by 25 bps after June 2026 meeting?", "fed-25-june",
       "2026-05-01T00:00:00Z", "2026-06-17T20:00:00Z", neg=True),
    _g(2, "Other", "fed-other-june", "2026-05-01T00:00:00Z", "2026-06-17T20:00:00Z", neg=True, other=True),
    _g(3, "Bitcoin Up or Down - June 3, 4PM ET", "bitcoin-up-or-down-june-3-4pm-et",
       "2026-06-03T19:00:00Z", "2026-06-03T20:00:00Z"),
    _g(4, "Will X win the primary?", "x-primary", "2026-03-01T00:00:00Z", "2026-06-02T00:00:00Z"),
    _g(5, "ECB cuts rates by 25 bps in June?", "ecb-june", "2026-04-01T00:00:00Z", "2026-06-05T00:00:00Z"),
    _g(6, "Lakers vs Celtics", "lal-bos", "2026-03-01T00:00:00Z", "2026-03-10T05:00:00Z",
       ev_start="2026-03-10T00:00:00Z"),
]


def test_filters_and_info_times():
    df, log = collect.filter_markets(RAW, min_life_days=2, exclude_slug_pattern=collect.RECURRING_CRYPTO)
    got = set(df["slug"])
    assert got == {"fed-25-june", "x-primary", "ecb-june", "lal-bos"}     # neg-risk kept by default
    assert log["dropped_neg_risk_placeholder"] == 1 and log["dropped_recurring_crypto"] == 1
    r = df.set_index("slug")
    assert r.loc["fed-25-june", "info_time_source"] == "fomc_statement"
    assert r.loc["fed-25-june", "info_time"] == pd.Timestamp("2026-06-17T18:00:00Z")   # 2pm EDT
    assert r.loc["ecb-june", "info_time_source"] == "scheduled_end"                    # not the Fed
    assert r.loc["lal-bos", "info_time_source"] == "event_start"


def test_placebo_include_pattern_and_prefer():
    df, _ = collect.filter_markets(RAW, include_slug_pattern="up-or-down", exclude_slug_pattern="")
    assert list(df["slug"]) == ["bitcoin-up-or-down-june-3-4pm-et"]
    pref = {f"0x{4:064x}"}
    df2, log = collect.filter_markets(RAW, n_markets=1, prefer_ids=pref, exclude_slug_pattern="")
    assert df2["condition_id"].iloc[0] == f"0x{4:064x}" and log["reused_downloaded"] == 1


def test_select_by_tags_unions_and_labels():
    class Fake(Client):
        def get(self, base, path, params=None):
            if path == "/tags/slug/fed":
                return {"id": "100", "slug": "fed"}
            if path == "/tags/slug/politics":
                return {"id": "200", "slug": "politics"}
            if path.startswith("/tags/slug/"):
                raise collect.APIError("404") if hasattr(collect, "APIError") else RuntimeError("404")
            if path.endswith("/tags"):
                mid = path.split("/")[2]
                return [{"id": "100"}] if mid == "1" else [{"id": "200"}]
            if path == "/markets/keyset":
                tid = params.get("tag_id")
                rows = [RAW[0], RAW[3]] if tid == "100" else [RAW[3]]   # market 4 in both tags
                return {"markets": rows, "next_cursor": None}
            raise AssertionError(path)

    df, log = collect.select_markets(Fake(), "2026-01-01", "2026-09-30", tags=["fed", "politics", "nope"],
                                     min_life_days=2)
    assert len(df) == 2 and log["by_tag"]["politics"]["new"] == 0
    assert set(df["tag"]) == {"fed"}


def test_load_cached_gamma(tmp_path):
    def save(name, url, body):
        (tmp_path / name).write_text(json.dumps({"url": url, "params": {}, "body": body}))
    save("a.json", "https://gamma-api.polymarket.com/markets/keyset", {"markets": RAW[:3], "next_cursor": "x"})
    save("b.json", "https://gamma-api.polymarket.com/markets", RAW[2:])            # legacy list, overlaps
    save("c.json", "https://gamma-api.polymarket.com/markets/1/tags", [{"id": "100"}])
    save("d.json", "https://data-api.polymarket.com/v2/trades", {"data": [{"x": 1}]})
    got = collect.load_cached_gamma([tmp_path])
    assert sorted(m["id"] for m in got) == [str(i) for i in range(1, 7)]
    df, _ = collect.filter_markets(got, include_slug_pattern="up-or-down", exclude_slug_pattern="")
    assert list(df["slug"]) == ["bitcoin-up-or-down-june-3-4pm-et"]


def test_select_from_old_market_list(tmp_path):
    # an older markets.parquet: no event columns, no info_time
    df, _ = collect.filter_markets(RAW, exclude_slug_pattern="")
    old = df.drop(columns=["event_id", "event_slug", "event_title", "neg_risk_other",
                           "event_start_time", "info_time", "info_time_source", "is_fed"])
    old.to_parquet(tmp_path / "markets.parquet")
    saved = collect.load_market_list(tmp_path / "markets.parquet")
    got, log = collect.filter_market_table(saved, include_slug_pattern="up-or-down", n_markets=5)
    assert list(got["slug"]) == ["bitcoin-up-or-down-june-3-4pm-et"]
    assert got["info_time_source"].iloc[0] == "scheduled_end" and log["from_saved_list"] == len(old)
