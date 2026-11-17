"""Live feed: new trades by likely-informed wallets -> site/data/live_feed.json.

The wallet list comes from history (run_flagging_v2.py --label-all), so nothing
here looks at the future: a wallet is only watched if its PAST results flagged it.

    # 1. one-off: find out which wallet filter the Data API v2 accepts (not documented in this repo)
    python live_feed.py --probe

    # 2. run it (polls every 60 s; Ctrl-C to stop)
    python live_feed.py --wallets results_v2/likely_informed_wallets_live.csv --top-n 150

    # if --probe finds no wallet filter, poll recent trades per open market instead
    python live_feed.py --mode market --max-markets 200

Open trades have no return until their market resolves; each cycle also checks
open trades' markets on Gamma and fills in return and PnL when they close.
Returns here ignore fees (the feed does not report them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from pm473 import clean, collect, config
from pm473.api import APIError, Client

WALLET_PARAMS = ["user", "proxy_wallet", "wallet", "address", "trader"]
EXTRA_ALIASES = {
    "title": ["title", "question", "market_title"],
    "outcome_label": ["outcome", "outcome_name"],
    "outcome_index": ["outcome_index", "outcomeIndex"],
    "slug": ["slug", "market_slug"],
    "event_slug": ["event_slug", "eventSlug"],
}
MAX_TRADES, MAX_SEEN = 500, 100_000


# --------------------------------------------------------------------------- parsing
def _pick(row: dict, names: list[str]):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


def normalize_row(row: dict) -> dict | None:
    """One raw Data API v2 trade row -> flat dict, or None if unusable."""
    g = {k: _pick(row, v) for k, v in {**clean.COLUMN_ALIASES, **EXTRA_ALIASES}.items()}
    try:
        price, size = float(g["price"]), float(g["size"])
        ts = float(g["timestamp"])
    except (TypeError, ValueError):
        return None
    if ts > 1e11:
        ts /= 1000.0
    side = str(g["side"] or "").upper()
    if g["wallet"] is None or g["condition_id"] is None or side not in ("BUY", "SELL") \
            or not (0 < price < 1) or size <= 0:
        return None
    oi = g["outcome_index"]
    return {"wallet": str(g["wallet"]).lower(), "condition_id": str(g["condition_id"]).lower(),
            "token_id": None if g["token_id"] is None else str(g["token_id"]),
            "side": side, "price": price, "size": size, "ts": int(ts),
            "tx_hash": str(g["tx_hash"] or ""), "title": g["title"], "outcome_label": g["outcome_label"],
            "outcome_index": int(oi) if oi is not None else None,
            "slug": g["slug"], "event_slug": g["event_slug"]}


def trade_key(t: dict) -> str:
    s = f'{t["tx_hash"]}|{t["wallet"]}|{t["token_id"]}|{t["side"]}|{t["price"]}|{t["size"]}|{t["ts"]}'
    return hashlib.sha1(s.encode()).hexdigest()[:20]


def held_return(side: str, price: float, size: float, outcome_index: int, winner_index: int):
    """Return and PnL (before fees) of one fill, held to resolution.
    BUY outcome o at p  -> long o at p;  SELL outcome o at p -> long (1-o) at 1-p."""
    long_idx = outcome_index if side == "BUY" else 1 - outcome_index
    entry = price if side == "BUY" else 1.0 - price
    payout = 1.0 if winner_index == long_idx else 0.0
    return (payout - entry) / entry, size * (payout - entry)


# --------------------------------------------------------------------------- API
def fetch_wallet(client, wallet: str, param: str, pages: int = 2):
    return list(client.paginate_v2("/v2/trades", {param: wallet, "limit": 100, "taker_only": False},
                                   max_pages=pages))


def fetch_market(client, condition_id: str, pages: int = 1):
    return list(client.paginate_v2("/v2/trades", {"condition": condition_id, "limit": 200,
                                                  "taker_only": False}, max_pages=pages))


def probe(client, wallet: str, params=WALLET_PARAMS) -> str | None:
    """Return the first query parameter that really filters /v2/trades by wallet."""
    for p in params:
        try:
            rows = list(client.paginate_v2("/v2/trades", {p: wallet, "limit": 5}, max_pages=1))
        except APIError as e:
            print(f"  {p}: rejected ({str(e)[:80]})")
            continue
        wallets = {str(_pick(r, clean.COLUMN_ALIASES["wallet"]) or "").lower() for r in rows}
        if rows and wallets <= {wallet.lower()}:
            print(f"  {p}: works ({len(rows)} rows, all for {wallet[:10]}...)")
            return p
        print(f"  {p}: accepted but not filtering ({len(rows)} rows, {len(wallets)} wallets)")
    return None


def market_info(client, cond: str, cache: dict) -> dict | None:
    """Gamma record for a condition id (cached). Uses collect.parse_market."""
    if cond in cache and cache[cond].get("closed_checked", 0) > time.time() - 300:
        return cache[cond]
    try:
        body = client.get(config.GAMMA_URL, "/markets", {"condition_ids": cond})
    except APIError:
        return cache.get(cond)
    rows = body if isinstance(body, list) else (body.get("data") or body.get("markets") or [])
    for m in rows:
        if str(m.get("conditionId", "")).lower() == cond:
            info = collect.parse_market(m) or {}
            cache[cond] = {"token_0": info.get("token_0"), "token_1": info.get("token_1"),
                           "winner_index": info.get("winner_index"), "question": info.get("question"),
                           "event_slug": info.get("event_slug"), "outcome_0": info.get("outcome_0"),
                           "outcome_1": info.get("outcome_1"), "closed": bool(m.get("closed")),
                           "end_time": None if pd.isna(info.get("end_time")) else str(info.get("end_time")),
                           "closed_checked": time.time()}
            return cache[cond]
    return cache.get(cond)


# --------------------------------------------------------------------------- state
def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"seen": [], "trades": [], "markets": {}, "param": None}


def save_state(path: Path, st: dict):
    st["seen"] = st["seen"][-MAX_SEEN:]
    st["trades"] = st["trades"][-MAX_TRADES:]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    tmp.replace(path)


def add_new(st: dict, rows: list[dict], wallets: set, since_ts: int) -> int:
    """Append unseen trades by watched wallets that are newer than `since_ts`."""
    seen, added = set(st["seen"]), 0
    for raw in rows:
        t = normalize_row(raw)
        if t is None or t["wallet"] not in wallets or t["ts"] < since_ts:
            continue
        k = trade_key(t)
        if k in seen:
            continue
        seen.add(k)
        st["seen"].append(k)
        t.update({"key": k, "status": "open", "return": None, "pnl": None})
        st["trades"].append(t)
        added += 1
    return added


def settle(client, st: dict):
    """Fill in market names, outcome labels and (when closed) return and PnL."""
    for t in st["trades"]:
        if t["status"] == "resolved":
            continue
        info = market_info(client, t["condition_id"], st["markets"])
        if not info:
            continue
        t["title"] = t.get("title") or info.get("question")
        t["event_slug"] = t.get("event_slug") or info.get("event_slug")
        if t.get("outcome_index") is None and t.get("token_id") is not None:
            t["outcome_index"] = (0 if t["token_id"] == info.get("token_0")
                                  else 1 if t["token_id"] == info.get("token_1") else None)
        if t["outcome_index"] is not None and not t.get("outcome_label"):
            t["outcome_label"] = info.get(f'outcome_{t["outcome_index"]}')
        if info.get("closed") and info.get("winner_index") is not None and t["outcome_index"] is not None:
            t["return"], t["pnl"] = held_return(t["side"], t["price"], t["size"],
                                                t["outcome_index"], info["winner_index"])
            t["status"] = "resolved"


def export(st: dict, path: Path):
    out = []
    for t in sorted(st["trades"], key=lambda x: -x["ts"])[:MAX_TRADES]:
        long_idx = None if t.get("outcome_index") is None else (t["outcome_index"] if t["side"] == "BUY"
                                                                else 1 - t["outcome_index"])
        label = t.get("outcome_label")
        held = f'{label}' if t["side"] == "BUY" and label else (f'against {label}' if label else t["side"])
        out.append({"wallet": t["wallet"], "time": pd.Timestamp(t["ts"], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                    "market": t.get("title") or t["condition_id"][:12] + "...", "held": held,
                    "entry_price": t["price"] if t["side"] == "BUY" else 1 - t["price"],
                    "shares": t["size"], "usd": t["price"] * t["size"], "return": t["return"], "pnl": t["pnl"],
                    "status": t["status"],
                    "link": f'https://polymarket.com/event/{t["event_slug"]}' if t.get("event_slug") else None})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
                               "trades": out}))
    tmp.replace(path)


# --------------------------------------------------------------------------- main
def cycle(client, st, wallets: list[str], mode: str, since_ts: int, max_markets: int) -> int:
    wset, added = set(wallets), 0
    if mode == "wallet":
        for w in wallets:
            try:
                added += add_new(st, fetch_wallet(client, w, st["param"]), wset, since_ts)
            except APIError as e:
                print(f"  {w[:10]}...: {str(e)[:100]}")
    else:
        mk = []
        for m in client.gamma_markets(closed="false", volume_num_min=50000):
            if m.get("conditionId"):
                mk.append((float(m.get("volumeNum") or 0), str(m["conditionId"]).lower()))
        for _, cond in sorted(mk, reverse=True)[:max_markets]:
            try:
                added += add_new(st, fetch_market(client, cond), wset, since_ts)
            except APIError as e:
                print(f"  {cond[:10]}...: {str(e)[:100]}")
    settle(client, st)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallets", default="results_v2/likely_informed_wallets_live.csv")
    ap.add_argument("--top-n", type=int, default=150, help="watch the N highest-evidence wallets")
    ap.add_argument("--mode", choices=["wallet", "market"], default="wallet")
    ap.add_argument("--max-markets", type=int, default=200)
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--interval", type=int, default=60, help="seconds between polls")
    ap.add_argument("--since-hours", type=float, default=24, help="on first run, ignore older trades")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--probe", action="store_true", help="find the wallet filter parameter and exit")
    a = ap.parse_args()

    client = Client()
    df = pd.read_csv(a.wallets).sort_values("Z", ascending=False).head(a.top_n)
    wallets = [w.lower() for w in df["wallet"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spath = out / "live_state.json"
    st = load_state(spath)

    if a.probe:
        p = probe(client, wallets[0])
        print("wallet filter parameter:", p or "none found -> use --mode market")
        return
    if a.mode == "wallet" and not st["param"]:
        st["param"] = probe(client, wallets[0])
        if not st["param"]:
            raise SystemExit("The Data API did not filter by wallet with any known parameter. "
                             "Run with --mode market instead.")
        save_state(spath, st)
    since = int(time.time() - a.since_hours * 3600) if not st["trades"] else 0
    print(f"watching {len(wallets)} wallets in {a.mode} mode; writing {out / 'live_feed.json'}")
    while True:
        n = cycle(client, st, wallets, a.mode, since, a.max_markets)
        since = 0
        save_state(spath, st)
        export(st, out / "live_feed.json")
        print(f"{time.strftime('%H:%M:%S')}  +{n} new trades  ({len(st['trades'])} kept, "
              f"{sum(t['status'] == 'open' for t in st['trades'])} open)")
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
