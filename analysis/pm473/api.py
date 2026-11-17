"""Thin, polite HTTP client for Gamma + Data API v2.

Every raw response page can be saved to disk (raw_dir) so the dataset is
reproducible and auditable -- useful when you write up the cleaning step.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from . import config


class APIError(RuntimeError):
    pass


class _Replay(RuntimeError):
    pass


class Client:
    def __init__(self, raw_dir: str | Path | None = None,
                 min_interval: float = config.MIN_SECONDS_BETWEEN_REQUESTS,
                 max_retries: int = 6, timeout: int = 30):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "ECE473-research/0.1"
        self.raw_dir = Path(raw_dir) if raw_dir else None
        if self.raw_dir:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last = 0.0

    # ------------------------------------------------------------------ core
    def get(self, base: str, path: str, params: dict | None = None):
        params = {k: _fmt(v) for k, v in (params or {}).items() if v is not None}
        url = base.rstrip("/") + "/" + path.lstrip("/")
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as e:
                _backoff(attempt, f"network error {e}")
                continue
            if r.status_code == 200:
                body = r.json()
                self._save(url, params, body)
                return body
            # v2 errors are JSON with a `retryable` flag
            retryable = r.status_code in (429, 500, 502, 503, 504)
            try:
                err = r.json()
                retryable = err.get("retryable", retryable)
            except ValueError:
                err = r.text[:300]
            if not retryable:
                raise APIError(f"{r.status_code} {url} {params} -> {err}")
            ra = r.headers.get("Retry-After")
            _backoff(attempt, f"{r.status_code} retrying", float(ra) if ra else None)
        raise APIError(f"gave up after {self.max_retries} tries: {url} {params}")

    def paginate_v2(self, path: str, params: dict, max_pages: int | None = None) -> Iterator[dict]:
        """Follow pagination.next_cursor. Original params are re-sent with the
        cursor (some routes 400 on a bare cursor)."""
        cursor, pages, seen = None, 0, set()
        while True:
            p = dict(params)
            if cursor:
                p["cursor"] = cursor
            body = self.get(config.DATA_URL, path, p)
            rows = body.get("data") or []
            if isinstance(rows, dict):  # single-object routes
                rows = [rows]
            yield from rows
            pages += 1
            cursor = (body.get("pagination") or {}).get("next_cursor")
            if not cursor or cursor in seen or (max_pages and pages >= max_pages):
                return
            seen.add(cursor)

    # ----------------------------------------------------------- endpoints
    def gamma_markets(self, page_size: int = 100, max_pages: int = 2000, **filters) -> Iterator[dict]:
        """All Gamma markets matching `filters`.

        1. /markets/keyset (cursor = `after_cursor`, <=100 per page).
        2. If keyset errors or replays a page, fall back to the legacy offset
           endpoint, splitting the end-date range into windows small enough to
           stay under its ~2,000-row offset cap.
        Markets are de-duplicated by id across both paths."""
        seen = set()
        try:
            for m in self._gamma_keyset(min(page_size, 100), max_pages, filters):
                if m.get("id") not in seen:
                    seen.add(m.get("id"))
                    yield m
            return
        except (APIError, _Replay) as e:
            print(f"  [gamma] keyset paging unavailable ({str(e)[:100]}); using date windows")
        for m in self._gamma_windowed(filters):
            if m.get("id") not in seen:
                seen.add(m.get("id"))
                yield m

    def _gamma_keyset(self, limit, max_pages, filters):
        cursor, seen_cursors, seen_ids = None, set(), set()
        for page in range(max_pages):
            params = {"limit": limit, **filters}
            if cursor:
                params["after_cursor"] = cursor
            body = self.get(config.GAMMA_URL, "/markets/keyset", params)
            rows = body.get("markets") or body.get("data") or [] if isinstance(body, dict) else body
            ids = {m.get("id") for m in rows}
            if page == 0 and rows and not _respects(rows, filters):
                raise _Replay("keyset ignored the filters")
            if page > 0 and rows and ids <= seen_ids:
                raise _Replay("keyset returned an already-seen page")
            seen_ids |= ids
            yield from rows
            cursor = body.get("next_cursor") if isinstance(body, dict) else None
            if not cursor or not rows:
                return
            if cursor in seen_cursors:
                raise _Replay("keyset cursor did not advance")
            seen_cursors.add(cursor)
            if page % 10 == 9:
                print(f"  [gamma] {len(seen_ids):,} markets listed so far")

    def _gamma_offset(self, filters, cap=2000, limit=100):
        """Legacy offset paging. Returns (rows, hit_cap)."""
        rows, offset = [], 0
        while True:
            if offset >= cap:
                return rows, True
            try:
                batch = self.get(config.GAMMA_URL, "/markets", {"limit": limit, "offset": offset, **filters})
            except APIError as e:
                if "offset too large" in str(e):
                    return rows, True
                raise
            if isinstance(batch, dict):
                batch = batch.get("data", [])
            if not batch:
                return rows, False
            rows += batch
            offset += len(batch)

    def _gamma_windowed(self, filters, min_window=pd.Timedelta(hours=6)):
        lo = pd.to_datetime(filters.get("end_date_min", "2020-01-01"), utc=True)
        hi = pd.to_datetime(filters.get("end_date_max", pd.Timestamp.now(tz="UTC")), utc=True)
        stack = [(lo, hi)]
        while stack:
            a, b = stack.pop()
            f = {**filters, "end_date_min": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "end_date_max": b.strftime("%Y-%m-%dT%H:%M:%SZ")}
            rows, hit_cap = self._gamma_offset(f)
            if hit_cap and (b - a) > min_window:
                mid = a + (b - a) / 2
                stack += [(mid, b), (a, mid)]       # split and redo this window
                continue
            if hit_cap:
                print(f"  [gamma] warning: window {a} to {b} still exceeds the offset cap; "
                      "some markets in it may be missing")
            print(f"  [gamma] {a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M}: {len(rows)} markets")
            yield from rows

    def trades_for_condition(self, condition_id: str, page_limit: int = 500,
                             taker_only: bool = False, max_pages: int | None = None) -> Iterator[dict]:
        """Full fill tape for one market. taker_only=False keeps maker-side rows,
        which PnL needs (both counterparties' inventory)."""
        yield from self.paginate_v2("/v2/trades", {
            "condition": condition_id, "limit": page_limit, "taker_only": taker_only,
        }, max_pages=max_pages)

    def price_history(self, token_id: str, interval: str = "max",
                      bucket_seconds: int | None = None) -> list[dict]:
        """Data API v2 price history (replaces CLOB /prices-history).
        Retention: 1-min >= 7 days, 5-min >= 60 days, 30-min >= 90 days,
        3h/12h permanent. Old markets therefore only have coarse prices --
        this project computes fine-grained prices from the trade tape instead."""
        return list(self.paginate_v2("/v2/prices-history", {
            "token_id": token_id, "interval": interval, "bucket_seconds": bucket_seconds,
        }))

    # ------------------------------------------------------------- helpers
    def _save(self, url, params, body):
        # Only metadata (Gamma) responses are archived. Trade pages are already
        # stored as per-market parquet files; archiving them too doubled disk use.
        if not self.raw_dir or config.GAMMA_URL not in url:
            return
        key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
        (self.raw_dir / f"{key}.json").write_text(
            json.dumps({"url": url, "params": params, "fetched_at": time.time(), "body": body}))


def _fmt(v):
    if isinstance(v, bool):          # v2 rejects Python-style True/False
        return "true" if v else "false"
    return v


def _backoff(attempt: int, why: str, retry_after: float | None = None):
    delay = retry_after if retry_after else min(60, 2 ** attempt)
    print(f"  [api] {why}; sleeping {delay:.1f}s")
    time.sleep(delay)


def _respects(rows, filters):
    """True if a page of Gamma markets looks filtered as requested."""
    ok = 0
    for m in rows:
        good = True
        if "closed" in filters:
            good &= bool(m.get("closed")) == (str(filters["closed"]).lower() == "true")
        if "volume_num_min" in filters:
            good &= float(m.get("volumeNum") or 0) >= float(filters["volume_num_min"]) * 0.99
        if "end_date_min" in filters and m.get("endDate"):
            good &= str(m["endDate"])[:10] >= str(filters["end_date_min"])[:10]
        if "end_date_max" in filters and m.get("endDate"):
            good &= str(m["endDate"])[:10] <= str(filters["end_date_max"])[:10]
        ok += good
    return ok >= 0.9 * len(rows)
