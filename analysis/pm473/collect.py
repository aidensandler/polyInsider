"""Step 1: pick a sample of resolved markets (Gamma) and download their trade tapes (Data API v2)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .api import Client
from .clean import COLUMN_ALIASES


# ----------------------------------------------------------------- markets
def _loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return None
    return x


def parse_market(m: dict) -> dict | None:
    """Flatten one Gamma market. Returns None if it isn't a two-token market.
    Each outcome of a multi-outcome ("neg-risk") event is its own two-token
    YES/NO market, so those pass too."""
    tokens = _loads(m.get("clobTokenIds")) or []
    outcomes = _loads(m.get("outcomes")) or []
    prices = _loads(m.get("outcomePrices")) or []
    if len(tokens) != 2 or len(prices) != 2:
        return None
    try:
        prices = [float(p) for p in prices]
    except (TypeError, ValueError):
        return None
    winner = int(np.argmax(prices)) if max(prices) >= 0.99 else None
    ev = (m.get("events") or [{}])[0] or {}
    return {
        "condition_id": (m.get("conditionId") or "").lower(),
        "market_id": m.get("id"),
        "question": m.get("question"),
        "slug": m.get("slug"),
        "event_id": ev.get("id"), "event_slug": ev.get("slug"), "event_title": ev.get("title"),
        "token_0": str(tokens[0]), "token_1": str(tokens[1]),
        "outcome_0": outcomes[0] if outcomes else "Yes",
        "outcome_1": outcomes[1] if len(outcomes) > 1 else "No",
        "value_0": prices[0], "value_1": prices[1],       # resolution payout per share
        "winner_index": winner,
        "neg_risk": bool(m.get("negRisk")),
        "neg_risk_other": bool(m.get("negRiskOther")),    # placeholder "Other" outcomes
        "protocol_version": m.get("version"),
        "start_time": pd.to_datetime(m.get("startDate") or m.get("createdAt"), utc=True, errors="coerce"),
        "end_time": pd.to_datetime(m.get("endDate"), utc=True, errors="coerce"),
        "closed_time": pd.to_datetime(m.get("closedTime"), utc=True, errors="coerce"),
        "event_start_time": pd.to_datetime(m.get("eventStartTime") or m.get("gameStartTime"),
                                           utc=True, errors="coerce"),
        "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        "uma_status": m.get("umaResolutionStatus"),
        "tag": m.get("_tag"),
    }


# Topic tags (Gamma slugs). Unknown slugs are skipped with a warning;
# run `python list_tags.py fed` etc. to see what exists.
DEFAULT_TAGS = ["politics", "elections", "us-election", "world-elections", "economy", "fed",
                "fed-rates", "inflation", "economic-policy", "macro-indicators"]
RECURRING_CRYPTO = r"up-or-down|updown|-up-down-|hit-price|above-on-|price-on-"


def load_cached_gamma(raw_dirs) -> list[dict]:
    """Market records from previously saved Gamma listing pages (no API calls).
    Every run saves its Gamma responses to <out>/raw/, so an earlier broad
    listing can be re-filtered for a new sample."""
    seen, out = set(), []
    for d in raw_dirs:
        for f in sorted(Path(d).glob("*.json")):
            with open(f) as fh:
                head = fh.read(300)
            if "gamma-api" not in head or "/markets" not in head or "/tags" in head:
                continue
            body = json.loads(Path(f).read_text()).get("body")
            rows = body.get("markets") or body.get("data") or [] if isinstance(body, dict) else body
            for m in rows or []:
                if isinstance(m, dict) and m.get("conditionId") and m.get("id") not in seen:
                    seen.add(m.get("id"))
                    out.append(m)
    return out


def resolve_tags(client: Client, slugs: list[str]) -> dict[str, str]:
    out = {}
    for sl in slugs:
        try:
            t = client.get(config.GAMMA_URL, f"/tags/slug/{sl}")
            if isinstance(t, dict) and t.get("id"):
                out[sl] = str(t["id"])
                continue
        except Exception:      # noqa: BLE001
            pass
        print(f"  [gamma] tag '{sl}' not found; skipping")
    return out


def check_tag_filter(client: Client, rows: list[dict], tag_id: str, slug: str, k: int = 5):
    """Spot-check that Gamma really applied tag_id (an ignored filter would
    silently return every market)."""
    import random
    sample = random.Random(0).sample(rows, min(k, len(rows)))
    hits = 0
    for m in sample:
        try:
            tags = client.get(config.GAMMA_URL, f"/markets/{m['id']}/tags")
            hits += any(str(t.get("id")) == tag_id for t in (tags or []))
        except Exception:      # noqa: BLE001
            pass
    if sample and hits < len(sample) / 2:
        print(f"  [gamma] WARNING: only {hits}/{len(sample)} spot-checked '{slug}' markets carry that "
              "tag directly (it may be on their event instead). Check the market list before downloading.")
    return hits, len(sample)


def select_markets(client: Client, end_min: str, end_max: str, min_volume: float = 50_000,
                   max_volume: float | None = None, include_neg_risk: bool = True,
                   n_markets: int | None = None, seed: int = 473, tags: list[str] | None = None,
                   min_life_days: float = 2.0, exclude_slug_pattern: str = RECURRING_CRYPTO,
                   include_slug_pattern: str = "", prefer_ids: set | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (markets, selection_log). Every exclusion is counted so the
    sampling frame can be reported honestly in the paper."""
    base = dict(closed=True, end_date_min=end_min, end_date_max=end_max,
                volume_num_min=min_volume, volume_num_max=max_volume)
    tag_log = {}
    if tags:
        raw, seen = [], set()
        for slug, tid in resolve_tags(client, tags).items():
            rows = list(client.gamma_markets(tag_id=tid, **base))
            check_tag_filter(client, rows, tid, slug)
            new = [dict(m, _tag=slug) for m in rows if m.get("id") not in seen]
            seen.update(m.get("id") for m in rows)
            raw += new
            tag_log[slug] = {"listed": len(rows), "new": len(new)}
            print(f"  [gamma] tag '{slug}': {len(rows):,} markets ({len(new):,} not already listed)")
    else:
        raw = list(client.gamma_markets(**base))
    df, log = filter_markets(raw, include_neg_risk=include_neg_risk, n_markets=n_markets, seed=seed,
                             end_min=end_min, end_max=end_max, min_volume=min_volume,
                             max_volume=max_volume, min_life_days=min_life_days,
                             exclude_slug_pattern=exclude_slug_pattern,
                             include_slug_pattern=include_slug_pattern, prefer_ids=prefer_ids)
    if tag_log:
        log["by_tag"] = tag_log
    return df, log


def filter_markets(raw: list[dict], include_neg_risk: bool = True, n_markets: int | None = None,
                   seed: int = 473, end_min=None, end_max=None, min_volume=None,
                   max_volume=None, min_life_days: float = 0.0, exclude_slug_pattern: str = "",
                   include_slug_pattern: str = "", prefer_ids: set | None = None) -> tuple[pd.DataFrame, dict]:
    log = {"gamma_returned": len(raw)}
    parsed = [parse_market(m) for m in raw]
    log["dropped_not_binary_or_unparseable"] = sum(p is None for p in parsed)
    df = pd.DataFrame([p for p in parsed if p is not None])
    return filter_market_table(df, log, include_neg_risk=include_neg_risk, n_markets=n_markets,
                               seed=seed, end_min=end_min, end_max=end_max, min_volume=min_volume,
                               max_volume=max_volume, min_life_days=min_life_days,
                               exclude_slug_pattern=exclude_slug_pattern,
                               include_slug_pattern=include_slug_pattern, prefer_ids=prefer_ids)


def load_market_list(path) -> pd.DataFrame:
    """A markets.parquet saved by an earlier run, with any columns that older
    versions didn't record filled in, ready for filter_market_table."""
    df = pd.read_parquet(path)
    defaults = {"event_id": None, "event_slug": None, "event_title": None, "neg_risk_other": False,
                "event_start_time": pd.NaT, "tag": None, "uma_status": None}
    for c, v in defaults.items():
        if c not in df:
            df[c] = v
    return df.drop(columns=[c for c in ("info_time", "info_time_source", "is_fed") if c in df])


def filter_market_table(df: pd.DataFrame, log: dict | None = None, include_neg_risk: bool = True,
                        n_markets: int | None = None, seed: int = 473, end_min=None, end_max=None,
                        min_volume=None, max_volume=None, min_life_days: float = 0.0,
                        exclude_slug_pattern: str = "", include_slug_pattern: str = "",
                        prefer_ids: set | None = None) -> tuple[pd.DataFrame, dict]:
    log = log if log is not None else {"from_saved_list": len(df)}
    if df.empty:
        return df, log
    df = df.copy()
    for c in ("start_time", "end_time", "closed_time", "event_start_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")   # all-NaT columns lose tz otherwise

    rt = df["closed_time"].fillna(df["end_time"])
    steps = [
        ("dropped_unresolved_or_split_resolution", df["winner_index"].notna()),
        ("dropped_protocol_v2", df["protocol_version"].isin(config.ALLOWED_PROTOCOL_VERSIONS)
                                | df["protocol_version"].isna()),
        ("dropped_missing_times", df["closed_time"].notna() | df["end_time"].notna()),
        ("dropped_neg_risk_placeholder", ~df["neg_risk_other"]),
    ]
    if not include_neg_risk:
        steps.append(("dropped_neg_risk", ~df["neg_risk"]))
    if exclude_slug_pattern:
        steps.append(("dropped_recurring_crypto",
                      ~df["slug"].fillna("").str.contains(exclude_slug_pattern, regex=True)))
    if include_slug_pattern:
        steps.append(("dropped_not_matching_include_pattern",
                      df["slug"].fillna("").str.contains(include_slug_pattern, regex=True)))
    if min_life_days:
        life = (rt - df["start_time"]).dt.total_seconds() / 86400
        steps.append(("dropped_shorter_than_min_life", life.fillna(0) >= min_life_days))
    # re-apply the request filters locally in case the API ignored any
    if end_min:
        steps.append(("dropped_outside_dates", (df["end_time"].fillna(rt) >= pd.Timestamp(end_min, tz="UTC"))
                      & (df["end_time"].fillna(rt) <= pd.Timestamp(end_max or "2100-01-01", tz="UTC")
                         + pd.Timedelta(days=1))))
    if min_volume:
        steps.append(("dropped_below_min_volume", df["volume"] >= min_volume))
    if max_volume:
        steps.append(("dropped_above_max_volume", df["volume"] <= max_volume))
    for name, mask in steps:            # sequential, so counts don't overlap
        mask = mask.reindex(df.index).fillna(False).astype(bool)
        log[name] = int((~mask).sum())
        df = df[mask]
    df = df.drop_duplicates("condition_id")
    df["resolve_time"] = df["closed_time"].fillna(df["end_time"])
    df["winner_index"] = df["winner_index"].astype(int)
    df = attach_info_times(df)

    log["eligible"] = len(df)
    log["eligible_neg_risk"] = int(df["neg_risk"].sum())
    if n_markets and len(df) > n_markets:
        # random, not "top by volume": a volume-ranked sample over-represents
        # headline markets where informed trading looks different.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(df))
        if prefer_ids:
            # markets already downloaded go first. This is unbiased only if
            # they were themselves a random draw from a superset of this pool.
            pref = df["condition_id"].isin(prefer_ids).to_numpy()
            order = np.concatenate([order[pref[order]], order[~pref[order]]])
            log["reused_downloaded"] = int(min(pref.sum(), n_markets))
        df = df.iloc[order[:n_markets]]
    log["selected"] = len(df)
    log["selected_neg_risk"] = int(df["neg_risk"].sum())
    log["selected_info_time_source"] = df["info_time_source"].value_counts().to_dict()
    return df.reset_index(drop=True), log


# US Fed only: generic "interest rate"/"bps" would also catch ECB, BoE, BoJ markets.
FED_PATTERN = r"\bfed\b|\bfomc\b|federal reserve|\bpowell\b|\bwarsh\b"


def attach_info_times(df: pd.DataFrame) -> pd.DataFrame:
    """When did the deciding information become public?

    * Fed decision markets: the official FOMC statement time (2:00 pm ET).
    * Markets with an event start (games, debates): that start time.
    * Otherwise: the scheduled end date, capped at the resolution time.
      For elections this is roughly polls closing; results arrive after,
      so post-`info_time` trading there is partly informed by counting.
    """
    df = df.copy()
    text = (df["question"].fillna("") + " " + df.get("event_title", pd.Series("", index=df.index)).fillna(""))
    is_fed = text.str.contains(FED_PATTERN, case=False, regex=True)
    fomc = pd.Series(pd.to_datetime(config.FOMC_STATEMENTS_UTC, utc=True))
    info = df["end_time"].where(df["end_time"] <= df["resolve_time"], df["resolve_time"])
    src = pd.Series("scheduled_end", index=df.index)
    has_ev = df["event_start_time"].notna() & (df["event_start_time"] <= df["resolve_time"])
    info, src = info.where(~has_ev, df["event_start_time"]), src.where(~has_ev, "event_start")
    for i in df.index[is_fed]:
        # the FOMC statement closest before (or within 3 days after) the market's resolution
        r = df.at[i, "resolve_time"]
        cand = fomc[(fomc <= r + pd.Timedelta(days=3)) & (fomc >= r - pd.Timedelta(days=10))]
        if len(cand):
            before = cand[cand <= r]
            info.at[i] = before.max() if len(before) else cand.min()
            src.at[i] = "fomc_statement"
    df["info_time"] = info.fillna(df["resolve_time"])
    df["info_time_source"] = src
    df["is_fed"] = is_fed
    return df


# ------------------------------------------------------------------ trades
# Only the trade fields the analysis uses (the feed also carries profile
# bios, image URLs and titles, which roughly triple memory per row).
RAW_KEEP = {n for names in COLUMN_ALIASES.values() for n in names}

def _fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, r = divmod(sec, 3600)
    m, _ = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"


class DownloadProgress:
    """One live status line: bar, percent of trading volume, markets done, rows fetched, elapsed time and a
    rough ETA. The ETA uses volume (a market's download time follows its trade count, which follows its
    volume) and counts the market currently downloading in proportion to the rows already fetched, using the
    rows-per-dollar seen on finished markets. It is still rough early on (a few huge markets dominate the
    total) and steadies after a few dozen markets. The line is cut to the terminal width so it redraws in
    place; when output is redirected to a file a plain line is printed every 30 seconds."""

    def __init__(self, n_total, vol_total, n_done, vol_done, stream=None, tty=None, width=None):
        self.n_total, self.vol_total = n_total, max(vol_total, 1.0)
        self.n_done, self.vol_done = n_done, vol_done
        self.vol_start = vol_done
        self.rows = 0                       # rows of finished markets this run
        self.vol_run = 0.0                  # volume of finished markets this run
        self.cur_rows, self.cur_vol, self.cur_id = 0, 0.0, ""
        self.t0 = time.time()
        self.stream = stream or sys.stdout
        self.tty = self.stream.isatty() if tty is None else tty
        self._width = width
        self._last = 0.0

    def start_market(self, cid, vol):
        self.cur_id, self.cur_vol, self.cur_rows = cid, float(vol), 0

    def effective_done(self) -> float:
        """Volume done, counting the market in progress by rows fetched so far."""
        done = self.vol_done
        if self.cur_rows and self.rows > 0 and self.vol_run > 0:
            done += min(self.cur_rows / (self.rows / self.vol_run), 0.95 * self.cur_vol)
        return done

    def line(self, width: int = 0) -> str:
        eff = self.effective_done()
        frac = min(1.0, eff / self.vol_total)
        bw = 16
        bar = "#" * int(bw * frac) + "-" * (bw - int(bw * frac))
        el = time.time() - self.t0
        rate = (eff - self.vol_start) / el if el > 5 else 0
        remaining = self.vol_total - eff
        eta = ("~" + _fmt_dur(remaining / rate)) if rate > 0 and remaining > 0 else ("done" if remaining <= 0 else "...")
        rows = self.rows + self.cur_rows
        txt = (f"[{bar}] {frac:5.1%} | {self.n_done:,}/{self.n_total:,} mkts | {rows:,} rows | {_fmt_dur(el)} | ETA {eta}"
               + (f" | {self.cur_id[:10]}… {self.cur_rows:,}" if self.cur_id and self.cur_rows else ""))
        return txt[:width] if width else txt

    def _term_width(self) -> int:
        if self._width:
            return self._width
        try:
            import shutil
            return max(40, shutil.get_terminal_size((100, 20)).columns - 2)
        except Exception:   # noqa: BLE001
            return 98

    def show(self, force=False):
        t = time.time()
        if not force and t - self._last < (0.3 if self.tty else 30):
            return
        self._last = t
        if self.tty:
            self.stream.write("\r\033[2K" + self.line(self._term_width()))
        else:
            self.stream.write(self.line() + "\n")
        self.stream.flush()

    def message(self, text):
        """Print a normal line above the progress line (for failures)."""
        self.stream.write(("\r\033[2K" if self.tty else "") + text + "\n")
        self.stream.flush()
        self.show(force=True)


def download_trades(client: Client, markets: pd.DataFrame, out_dir: str | Path,
                    page_limit: int = 500, progress: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """One parquet per market. Resumable: markets already on disk are skipped,
    files are written atomically (tmp then rename), and a market that fails
    is logged and skipped instead of stopping the run. Re-running retries it.
    Shows a live progress line (see DownloadProgress) and appends one line per finished market to
    <out_dir>/_download_log.txt."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    cids = list(markets["condition_id"])
    vols = (markets["volume"].fillna(0).astype(float).to_numpy() if "volume" in markets else np.ones(len(cids)))
    done_flags = [(out_dir / f"{c}.parquet").exists() for c in cids]
    prog = DownloadProgress(len(cids), float(vols.sum()), int(sum(done_flags)),
                            float(sum(v for v, d in zip(vols, done_flags) if d)))
    log = open(out_dir / "_download_log.txt", "a")
    if progress:
        print(f"  {prog.n_done:,} of {len(cids):,} markets already on disk; {len(cids) - prog.n_done:,} to download")
        prog.show(force=True)
    for i, (cid, vol) in enumerate(zip(cids, vols)):
        f = out_dir / f"{cid}.parquet"
        if f.exists():
            continue
        prog.start_market(cid, vol)
        try:
            rows = []
            for r in client.trades_for_condition(cid, page_limit=page_limit):
                rows.append(r)
                if progress and len(rows) % 500 == 0:
                    prog.cur_rows = len(rows)
                    prog.show()
            df = pd.DataFrame(rows)
            df = df[[c for c in df.columns if c in RAW_KEEP]]   # drop bios, images, titles
            df["_condition_query"] = cid
            tmp = f.with_suffix(".tmp")
            df.to_parquet(tmp, index=False)
            tmp.rename(f)
            prog.n_done += 1
            prog.vol_done += float(vol)
            prog.vol_run += float(vol)
            prog.rows += len(df)
            prog.cur_rows, prog.cur_id = 0, ""
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  [{i+1}/{len(cids)}] {cid}  {len(df):,} rows\n"); log.flush()
            if progress:
                prog.show()
        except Exception as e:      # noqa: BLE001 -- keep going, retry on next run
            failed.append(cid)
            msg = f"  [{i+1}/{len(cids)}] {cid[:10]}… FAILED ({type(e).__name__}: {str(e)[:120]})"
            log.write(time.strftime('%Y-%m-%d %H:%M:%S') + msg + "\n"); log.flush()
            if progress:
                prog.message(msg)
            else:
                print(msg)
    log.close()
    if progress:
        prog.show(force=True)
        prog.stream.write("\n")
        prog.stream.flush()
    files = [out_dir / f"{c}.parquet" for c in cids if (out_dir / f"{c}.parquet").exists()]
    return files, failed
