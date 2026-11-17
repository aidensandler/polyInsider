"""Live alerts: score new trades in open markets with the alert model -> site/data/live_alerts.json.

Needs results_v2/alerts/live_model.joblib and wallet_snapshot.parquet (python evaluate_alerts.py --live-model).
A trade is scored with only what is known at that moment: the wallet's snapshot of past results, the price paid,
the size, and the hours to the market's scheduled end. Two kinds of alert:
  * model alert: predicted edge per share above the threshold saved with the model;
  * past-score rule alert: wallet past score Z >= 3 with at least 10 past events.

    python live_alerts.py                      # polls the largest open markets every 60 s; Ctrl-C to stop
    python live_alerts.py --once               # one pass
    python live_alerts.py --max-markets 100

This is a demonstration. The model was validated on historical data only (see the Alerts page), the Data API polling
is shared with live_feed.py, and open trades have no return until their market resolves (returns ignore fees).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import live_feed as lf
from pm473 import alerts as al
from pm473.api import APIError, Client

MIN_USD, MAX_ALERTS = 5.0, 600
RULE_Z, RULE_MIN_EVENTS = 3.0, 10


def load_model(res: Path):
    import joblib
    model = joblib.load(res / "live_model.joblib")
    info = json.loads((res / "live_model_info.json").read_text())
    snap = pd.read_parquet(res / "wallet_snapshot.parquet").drop_duplicates("address").set_index("address")
    return model, info["threshold"], snap


def trade_features(t: dict, info: dict, w: pd.Series) -> dict | None:
    """Feature row for one live trade (t from live_feed.normalize_row, info from live_feed.market_info,
    w = the wallet's snapshot row). None if the held outcome is unknown."""
    oi = t.get("outcome_index")
    if oi is None:
        return None
    buy = t["side"] == "BUY"
    held = oi if buy else 1 - oi
    entry = t["price"] if buy else 1.0 - t["price"]
    end = pd.to_datetime(info.get("end_time"), utc=True, errors="coerce") if info else pd.NaT
    hours = (end - pd.Timestamp(t["ts"], unit="s", tz="UTC")).total_seconds() / 3600 if pd.notna(end) else np.nan
    row = {"entry_price": entry, "held_outcome": held, "log_usdc": np.log1p(t["price"] * t["size"]),
           "log_hours_to_sched": np.log1p(max(hours, 0)) if np.isfinite(hours) else np.nan,
           "tag_code": np.nan}                       # the live market record has no topic tag; the model treats it as missing
    for k in al.WALLET_FEATURES:
        row[k] = float(w[k])
    return row


def score_trade(model, threshold: float, row: dict) -> dict:
    x = pd.DataFrame([row])[al.FEATURES]
    s = float(model.predict(x)[0])
    return {"score": s, "model_alert": s >= threshold,
            "rule_alert": bool(row["Z_past"] >= RULE_Z and row["n_past"] >= RULE_MIN_EVENTS)}


def process_rows(rows: list[dict], st: dict, snap: pd.DataFrame, model, threshold: float, since_ts: int, client=None) -> int:
    """Score unseen trades by wallets in the snapshot; keep the ones that raise an alert."""
    seen, added = set(st["seen"]), 0
    for raw in rows:
        t = lf.normalize_row(raw)
        if t is None or t["ts"] < since_ts or t["wallet"] not in snap.index or t["price"] * t["size"] < MIN_USD:
            continue
        k = lf.trade_key(t)
        if k in seen:
            continue
        seen.add(k); st["seen"].append(k)
        info = lf.market_info(client, t["condition_id"], st["markets"]) if client is not None else st["markets"].get(t["condition_id"])
        if info and t.get("outcome_index") is None and t.get("token_id") is not None:
            t["outcome_index"] = 0 if t["token_id"] == info.get("token_0") else 1 if t["token_id"] == info.get("token_1") else None
        w = snap.loc[t["wallet"]]
        feats = trade_features(t, info, w)
        if feats is None:
            continue
        sc = score_trade(model, threshold, feats)
        if not (sc["model_alert"] or sc["rule_alert"]):
            continue
        t.update({"key": k, "status": "open", "return": None, "pnl": None, "score": sc["score"], "model_alert": sc["model_alert"],
                  "rule_alert": sc["rule_alert"], "past_score": float(w["Z_past"]), "past_events": int(w["n_past"])})
        st["trades"].append(t); added += 1
    return added


def export(st: dict, path: Path):
    out = []
    for t in sorted(st["trades"], key=lambda x: -x["ts"])[:MAX_ALERTS]:
        buy = t["side"] == "BUY"
        label = t.get("outcome_label")
        out.append({"wallet": t["wallet"], "time": pd.Timestamp(t["ts"], unit="s", tz="UTC").strftime("%Y-%m-%d %H:%M"),
                    "market": t.get("title") or t["condition_id"][:12] + "...", "held": (label if buy and label else (f"against {label}" if label else t["side"])),
                    "entry_price": t["price"] if buy else 1 - t["price"], "usd": t["price"] * t["size"], "score": t["score"],
                    "past_score": t["past_score"], "past_events": t["past_events"], "model_alert": t["model_alert"], "rule_alert": t["rule_alert"],
                    "status": t["status"], "return": t["return"], "pnl": t["pnl"],
                    "link": f'https://polymarket.com/event/{t["event_slug"]}' if t.get("event_slug") else None})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"updated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"), "alerts": out}))
    tmp.replace(path)


def cycle(client, st, snap, model, threshold, since_ts, max_markets) -> int:
    mk, added = [], 0
    for m in client.gamma_markets(closed="false", volume_num_min=50000):
        if m.get("conditionId"):
            mk.append((float(m.get("volumeNum") or 0), str(m["conditionId"]).lower()))
    for _, cond in sorted(mk, reverse=True)[:max_markets]:
        try:
            added += process_rows(lf.fetch_market(client, cond), st, snap, model, threshold, since_ts, client)
        except APIError as e:
            print(f"  {cond[:10]}...: {str(e)[:100]}")
    lf.settle(client, st)
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_v2/alerts")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--max-markets", type=int, default=150)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--since-hours", type=float, default=24)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    model, threshold, snap = load_model(Path(a.results))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spath = out / "live_alerts_state.json"
    st = lf.load_state(spath)
    client, since = Client(), (int(time.time() - a.since_hours * 3600) if not st["trades"] else 0)
    print(f"{len(snap):,} wallets in the snapshot; model alert threshold {threshold:.3f}; writing {out / 'live_alerts.json'}")
    while True:
        n = cycle(client, st, snap, model, threshold, since, a.max_markets)
        since = 0
        lf.save_state(spath, st)
        export(st, out / "live_alerts.json")
        print(f"{time.strftime('%H:%M:%S')}  +{n} new alerts  ({len(st['trades'])} kept, {sum(t['status'] == 'open' for t in st['trades'])} open)")
        if a.once:
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
