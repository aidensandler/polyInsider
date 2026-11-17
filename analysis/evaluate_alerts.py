"""Step 2 of the alert model: train, score the held-out period, and test 'what if you traded with the alerts'.

    python build_alert_candidates.py                      # step 1 (once)
    python evaluate_alerts.py                             # main cut-off 2026-06-01 (about 3 min)
    python evaluate_alerts.py --cutoff 2026-05-01         # extra cut-offs for robustness (one per run)
    python evaluate_alerts.py --cutoff 2026-07-01
    python evaluate_alerts.py --merge                     # combine the per-cut-off tables
    python evaluate_alerts.py --live-model                # fit on all resolved history for live scoring

Per cut-off the models train on trades made BEFORE it (in events resolved by it) and are tested on trades
made AFTER it. Four models are compared so the value of wallet history can be separated from price effects:
  all features | trade features only (no wallet history) | wallet history only | price only.
Strategies copied with a flat $100 stake after a delay of 0, 5 minutes and 1 hour:
  the four models' alerts (top ~1% of validation-period scores), trading with the BH-flagged wallets,
  a dynamic rule (past score Z >= 3 with at least 10 past events), and randomly chosen trades.
Outputs go to results_v2/alerts/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pm473 import alert_backtest as bt
from pm473 import alerts as al
from pm473 import skill_test as st

MODELS = {"All features": al.FEATURES, "Trade features only": al.TRADE_FEATURES,
          "Wallet history only": al.WALLET_FEATURES, "Price only": al.PRICE_FEATURES}
DELAYS = {"0 min (upper bound)": 0, "5 min": 300, "1 hour": 3600}
STAKE = 100.0


def deciles(te: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    d = te.assign(score=s)
    d["decile"] = pd.qcut(d["score"].rank(method="first"), 10, labels=False) + 1
    g = d.groupby("decile")
    out = pd.DataFrame({"trades": g.size(), "mean_predicted_edge": g["score"].mean(), "mean_realized_edge": g["edge"].mean(),
                        "mean_price": g["entry_price"].mean(), "events": g["event_key"].nunique()})
    out["roi_dollar_weighted"] = (d.assign(p=d["edge"] * d["size"], k=d["entry_price"] * d["size"]).groupby("decile")["p"].sum()
                                  / d.assign(k=d["entry_price"] * d["size"]).groupby("decile")["k"].sum())
    return out.reset_index()


MODELS_CTX = {"All features": al.FEATURES, "Price only": al.PRICE_FEATURES,
              "All + bot features": al.FEATURES + al.BOT_FEATURES,
              "All + behaviour": al.FEATURES + al.BEHAVIOUR_FEATURES,
              "All + agreement": al.FEATURES + al.AGREE_FEATURES,
              "All + all context": al.FEATURES + al.CONTEXT_FEATURES}


MODELS_WINDOWS = {"All features": al.FEATURES, "Price only": al.PRICE_FEATURES,
                  "Crowd flow (no wallet skill)": al.TRADE_FEATURES + al.FLOW_FEATURES,
                  "Trade + agreement (no own history)": al.TRADE_FEATURES + al.AGREE_FEATURES,
                  "All + agreement": al.FEATURES + al.AGREE_FEATURES,
                  "All + agreement + flow": al.FEATURES + al.AGREE_FEATURES + al.FLOW_FEATURES}


def run_cutoff(c, ev, mk, data, cutoff, out, main: bool, n_boot=500, n_draws=200, models=None, ctx=False,
               test_end=None, label=None, collect=None):
    """test_end: only trades made before it, in events that resolved before it (a forward window [cutoff, test_end)).
    label: name for output files (default: the cutoff). collect: dict that receives per-alert results (no extra
    slippage) keyed by (strategy, delay), used to pool several windows."""
    models = models or MODELS
    label = label or cutoff
    tr, te = al.split_train_test(c, cutoff)
    if test_end is not None:
        end = pd.Timestamp(test_end)
        end = end if end.tzinfo else end.tz_localize("UTC")
        te = te[(te["ts"] < end) & (te["event_time"] < end)]
    print(f"[{label}] train {len(tr):,} trades, test {len(te):,} trades", flush=True)
    tape = bt.make_tape(data / "fills_clean.parquet", te["condition_id"].unique(), cutoff)
    rows, info, scores, alerts_by = [], {"cutoff": label, "train_trades": len(tr), "test_trades": len(te)}, {}, {}
    bot_cut = None
    if ctx:      # 'bot-like' = most two-sided 10% of qualifying trades in the TRAINING period (fixed before test)
        bot_cut = {"lw_two_sided_past": float(tr["lw_two_sided_past"].quantile(0.90)),
                   "lw_trades_per_mkt_past": float(tr["lw_trades_per_mkt_past"].quantile(0.90))}
        info["bot_cut"] = bot_cut
    for name, feats in models.items():
        model, mi = al.train_alert_model(tr, features=feats, iter_grid=(40, 100))
        s = al.score(model, te, feats)
        scores[name] = s
        alerts_by[name] = bt.dedupe_alerts(te[s >= mi["threshold"]])
        info[name] = {"rounds": mi["rounds"], "threshold": mi["threshold"], "alert_rate_test": float((s >= mi["threshold"]).mean()),
                      "alerts_deduped": int(len(alerts_by[name]))}
        print(f"  {name}: rounds {mi['rounds']}, threshold {mi['threshold']:.3f}, alerts {len(alerts_by[name]):,}", flush=True)
        if main and name == "All features":
            joblib_dump(model, out / "model_main.joblib")
    if ctx:      # idea 1 as a hard filter: drop alerts from bot-like wallets
        botlike = ((te["lw_two_sided_past"] >= bot_cut["lw_two_sided_past"])
                   | (te["lw_trades_per_mkt_past"] >= bot_cut["lw_trades_per_mkt_past"])).to_numpy()
        info["test_share_botlike"] = float(botlike.mean())
        for base in ("All features", "All + all context"):
            s = scores[base]
            thr = info[base]["threshold"]
            alerts_by[f"{base}, bot-like wallets removed"] = bt.dedupe_alerts(te[(s >= thr) & ~botlike])
        # agreement as a simple rule: at least 2 other good traders on the same side in the last hour
        alerts_by["Good wallet + 2 others agree (1h)"] = bt.dedupe_alerts(te[(te["is_good"] > 0) & (te["agree_n_1h"] >= 2)])
    # strategies that need no model
    form = ev[ev["period_time"] < pd.Timestamp(cutoff, tz="UTC")]
    flagged = st.flag_wallets(st.wallet_stats(form, 10), q=0.10, method="BH")
    codes = set(flagged.index[flagged["informed"]])
    alerts_by["Trade with flagged wallets"] = bt.dedupe_alerts(te[te["wallet"].isin(codes)])
    alerts_by["Past score Z >= 3 (10+ events)"] = bt.dedupe_alerts(te[(te["Z_past"] >= 3) & (te["n_past"] >= 10)])
    info["flagged_wallets"] = len(codes)
    pool = bt.dedupe_alerts(te.sample(min(len(te), 600_000), random_state=0))
    for slip in (0.0, 0.01):
        for dlab, d in DELAYS.items():
            if slip and d == 0:
                continue
            for name, a in alerts_by.items():
                r = bt.follower_results(a, tape, d, STAKE, slip)
                if collect is not None and slip == 0:
                    collect.setdefault((name, dlab), []).append(
                        r[["wallet", "event_key", "pnl", "stake", "payout", "ret", "p_exec"]].assign(window=label))
                sm = bt.summarize(r, n_boot)
                rb = bt.random_benchmark(pool, tape, max(1, min(len(a), len(pool) // 2)), d, slip, n_draws) if len(a) else {}
                rows.append({"cutoff": label, "strategy": name, "delay": dlab, "delay_s": d, "slippage": slip, **sm,
                             "random_mean": rb.get("mean"), "random_p05": rb.get("p05"), "random_p95": rb.get("p95")})
            rr = bt.follower_results(pool, tape, d, STAKE, slip)
            if collect is not None and slip == 0:
                collect.setdefault(("All candidate trades", dlab), []).append(
                    rr[["wallet", "event_key", "pnl", "stake", "payout", "ret", "p_exec"]].assign(window=label))
            rows.append({"cutoff": label, "strategy": "All candidate trades", "delay": dlab, "delay_s": d, "slippage": slip,
                         **bt.summarize(rr, n_boot)})
    bt_table = pd.DataFrame(rows)
    parts = out / "parts"; parts.mkdir(exist_ok=True, parents=True)
    bt_table.to_csv(parts / f"backtest_{label}.csv", index=False)
    json.dump(info, open(parts / f"info_{label}.json", "w"), indent=1, default=float)

    # model quality + ablation (value of wallet history beyond price)
    q = pd.concat({k: deciles(te, s) for k, s in scores.items()}, names=["model"]).reset_index(level=0)
    q.to_csv(parts / f"deciles_{label}.csv", index=False)

    if main:
        ptime = st.period_times(mk)
        ek = st.event_keys(mk)
        cm = mk.set_index("condition_id")
        # replay of the alerts at a 5 minute delay, with outcomes (main model, and the simple past-score rule)
        model_main = joblib_load(out / "model_main.joblib")
        for key, fname in (("All features", "replay_alerts.parquet"), ("Past score Z >= 3 (10+ events)", "replay_rule_alerts.parquet")):
            r = bt.follower_results(alerts_by[key], tape, 300, STAKE, 0.0)
            r["score"] = al.score(model_main, r)
            r["market"] = r["condition_id"].map(cm["question"])
            r["held_label"] = np.where(r["held_outcome"] == 0, r["condition_id"].map(cm["outcome_0"]), r["condition_id"].map(cm["outcome_1"]))
            r["event_title"] = r["condition_id"].map(cm["event_title"]); r["event_slug"] = r["condition_id"].map(cm["event_slug"])
            r["resolved_at"] = r["event_key"].map(ptime)
            r.to_parquet(out / fname, index=False)
        # cumulative PnL by resolution day for the headline strategies
        curves = []
        for name in ("All features", "Price only", "Trade with flagged wallets"):
            rr = bt.follower_results(alerts_by[name], tape, 300, STAKE, 0.0)
            cur = bt.pnl_curve(rr, ptime); cur["strategy"] = name; curves.append(cur)
        pd.concat(curves).to_csv(out / "cumulative_pnl.csv", index=False)
        # lead-time sample: how long before resolution each group trades
        samp = []
        def take(df, g, n=None):
            d = df if n is None or len(df) <= n else df.sample(n, random_state=1)
            samp.append(d[["wallet", "event_key", "hours_to_resolve", "log_hours_to_sched", "usdc", "edge", "payout", "entry_price", "size"]].assign(group=g))
        take(alerts_by["All features"], "Model alerts")
        take(te[te["wallet"].isin(codes)], "Flagged wallets' trades")
        take(alerts_by["Past score Z >= 3 (10+ events)"], "Past score Z >= 3 alerts")
        take(te, "All candidate trades", 150_000)
        pd.concat(samp).to_parquet(out / "lead_time_sample.parquet", index=False)
        json.dump(info, open(out / "run_info_main.json", "w"), indent=1, default=float)
    print(f"[{cutoff}] done", flush=True)


def joblib_dump(m, path):
    import joblib
    joblib.dump(m, path)


def joblib_load(path):
    import joblib
    return joblib.load(path)


def merge(out: Path):
    parts = out / "parts"
    for stem in ("backtest", "deciles"):
        fs = sorted(parts.glob(f"{stem}_*.csv"))
        if fs:
            frames = []
            for f in fs:
                df = pd.read_csv(f)
                if "cutoff" not in df:
                    df.insert(0, "cutoff", f.stem.split("_", 1)[1])
                frames.append(df)
            pd.concat(frames).to_csv(out / f"{stem}_table.csv" if stem == "backtest" else out / "model_deciles.csv", index=False)
    infos = {f.stem.replace("info_", ""): json.load(open(f)) for f in sorted(parts.glob("info_*.json"))}
    json.dump(infos, open(out / "run_info.json", "w"), indent=1)
    print("merged:", [p.name for p in sorted(out.glob("*.csv"))])


def live_model(c, ev, mk, data, out):
    """Fit on every resolved trade (labels known) and save the model with a per-wallet snapshot of history to date."""
    done = c[c["event_time"] <= c["ts"].max()]
    model, mi = al.train_alert_model(done, iter_grid=(40, 100))
    joblib_dump(model, out / "live_model.joblib")
    h = al.wallet_history(ev).sort_values("hist_time").groupby("wallet").tail(1)
    names = pd.read_parquet(data / "wallet_codes.parquet")
    names = names.set_index("code")["wallet"] if "code" in names.columns else names["wallet"]
    snap = pd.DataFrame({"address": h["wallet"].map(names).str.lower(), "n_past": h["n_past"],
                         "Z_past": h["sum_z"] / np.sqrt(h["n_past"].clip(lower=1)), "mean_z_past": h["sum_z"] / h["n_past"].clip(lower=1),
                         "ppd_past": h["sum_c"] / h["sum_vol"].where(h["sum_vol"] > 0), "win_rate_past": h["wins"] / h["n_past"].clip(lower=1),
                         "log_vol_past": np.log1p(h["sum_vol"]), "last_z": h["last_z"]})
    snap = snap[snap["n_past"] >= 3].dropna(subset=["address"])
    snap.to_parquet(out / "wallet_snapshot.parquet", index=False)
    json.dump({"threshold": mi["threshold"], "rounds": mi["rounds"], "train_rows": mi["train_rows"], "wallets_in_snapshot": int(len(snap)),
               "fitted_through": str(c["ts"].max())}, open(out / "live_model_info.json", "w"), indent=1)
    print("live model saved;", len(snap), "wallets in snapshot; threshold", round(mi["threshold"], 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events/live")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--cutoff", default="2026-06-01")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--live-model", action="store_true")
    ap.add_argument("--ctx", action="store_true", help="add bot, behaviour and agreement features (ideas 1-3)")
    ap.add_argument("--ctx-out", default="alerts_ctx", help="output folder under --results for --ctx runs")
    ap.add_argument("--candidates", default=None, help="candidates folder under --results (default alerts)")
    a = ap.parse_args()
    out = Path(a.results) / (a.candidates or "alerts")
    if a.merge:
        merge(Path(a.results) / (a.ctx_out if a.ctx else "alerts"))
        return
    c = pd.read_parquet(out / "candidates.parquet")
    ev = pd.read_parquet(Path(a.results) / "events_table_events.parquet")
    mk = pd.read_parquet(Path(a.data) / "markets_coded.parquet")
    if a.live_model:
        live_model(c, ev, mk, Path(a.data), out)
        return
    if a.ctx:                                         # ideas 1-3; separate output folder, nothing existing is overwritten
        c = al.add_context_features(c)
        out = Path(a.results) / a.ctx_out
        out.mkdir(parents=True, exist_ok=True)
        run_cutoff(c, ev, mk, Path(a.data), a.cutoff, out, main=False, models=MODELS_CTX, ctx=True)
        return
    run_cutoff(c, ev, mk, Path(a.data), a.cutoff, out, main=(a.cutoff == "2026-06-01"))


if __name__ == "__main__":
    main()
