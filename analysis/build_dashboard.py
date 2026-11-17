"""Build the Polyinsider dashboard data (site/data/data.json) from saved results.

    python run_flagging_v2.py --data data_events/live --label-all      # live labels + scores
    python evaluate_model.py                                           # validation numbers/charts
    python build_dashboard.py --data data_events/live --results results_v2
    python -m http.server -d site 8000        # then open http://localhost:8000

Shows the flagged wallets AND the highest-scoring wallets that just missed the cut-off,
each clearly labelled. Writes only under site/data/ (data.json); never touches data*/.
Wording on the page is "likely informed": the evidence shows persistent out-of-sample
outperformance, not proof of private information.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pm473 import trade_returns as tr

COLS = ["wallet", "condition_id", "ts", "size", "p0", "dir0", "usdc", "fee_usdc"]


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return str(o)


def _histogram(z: np.ndarray, lo=-5.0, hi=12.0, width=0.5):
    edges = np.arange(lo, hi + width, width)
    counts, _ = np.histogram(np.clip(z, lo, hi - 1e-9), bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist(), "n": int(len(z))}


def _read_csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


def build(data_dir: Path, results: Path, out: Path, top_n: int, trades_per_wallet: int, scores_file: str):
    scores = pd.read_csv(results / scores_file)
    scores = scores[scores["eligible"]].sort_values("Z_adj", ascending=False)
    flagged_n = int(scores["informed"].sum())
    cutoff = float(scores.loc[scores["informed"], "Z_adj"].min()) if flagged_n else None
    show = pd.concat([scores[scores["informed"]], scores[~scores["informed"]].head(max(0, top_n - flagged_n))])
    show = show.sort_values("Z_adj", ascending=False)

    codes = pd.read_parquet(data_dir / "wallet_codes.parquet")
    addr2code = dict(zip(codes["wallet"], codes["code"]))
    show["code"] = show["wallet"].map(addr2code)
    show = show.dropna(subset=["code"])
    want = show["code"].astype(int).tolist()

    mk = pd.read_parquet(data_dir / "markets_coded.parquet")
    avail = [c for c in COLS if c in pq.read_schema(data_dir / "fills_clean.parquet").names]
    fills = pq.read_table(data_dir / "fills_clean.parquet", columns=avail,
                          filters=[("wallet", "in", want)]).to_pandas()
    t = tr.add_trade_returns(fills, mk)
    summ = tr.wallet_trade_summary(t)
    mkx = mk.set_index("condition_id")
    t["question"] = t["condition_id"].map(mkx["question"])
    t["event_slug"] = t["condition_id"].map(mkx["event_slug"]) if "event_slug" in mkx else None
    lab0, lab1 = t["condition_id"].map(mkx["outcome_0"]), t["condition_id"].map(mkx["outcome_1"])
    t["held"] = np.where(t["held_outcome"] == 0, lab0, lab1)
    t["ts"] = pd.to_datetime(t["ts"], utc=True)
    t = t.sort_values("ts", ascending=False)
    by_wallet = {k: v.head(trades_per_wallet) for k, v in t.groupby("wallet")}

    wallets = []
    for _, w in show.iterrows():
        code = int(w["code"])
        s = summ.loc[code] if code in summ.index else None
        tw = by_wallet.get(code)
        trades = []
        if tw is not None:
            for r in tw.itertuples():
                trades.append({
                    "time": r.ts.strftime("%Y-%m-%d %H:%M"), "market": r.question, "held": r.held,
                    "entry_price": _f(r.entry_price), "shares": _f(r.size), "usd": _f(r.usdc),
                    "return": _f(r.trade_return) if r.resolved else None,
                    "pnl": _f(r.trade_pnl) if r.resolved else None,
                    "status": "resolved" if r.resolved else "open",
                    "link": f"https://polymarket.com/event/{r.event_slug}" if isinstance(r.event_slug, str) else None})
        wallets.append({
            "address": w["wallet"], "flagged": bool(w["informed"]),
            "score": _f(w["Z_adj"]), "p_value": _f(w["p"]), "events_traded": int(w["n_events"]),
            "total_pnl": _f(s["total_pnl"]) if s is not None else None,
            "pnl_per_dollar": _f(s["pnl_per_dollar"]) if s is not None else None,
            "mean_trade_return": _f(s["mean_trade_return"]) if s is not None else None,
            "median_trade_return": _f(s["median_trade_return"]) if s is not None else None,
            "win_rate": _f(s["win_rate"]) if s is not None else None,
            "resolved_trades": int(s["resolved_trades"]) if s is not None else 0,
            "trades": trades})

    # ---- validation numbers and chart data (from run_flagging_v2 / evaluate_model)
    perf = {}
    pj = results / "eval" / "model_performance.json"
    if pj.exists():
        perf = json.loads(pj.read_text())
    ev_m = (perf.get("events") or {}).get("single_split", {})
    pl_m = (perf.get("placebo") or {}).get("single_split", {})
    deciles = {}
    for name in ("events", "placebo"):
        d = _read_csv(results / "eval" / f"decile_table_{name}.csv")
        if d is not None:
            deciles[name] = d[["decile", "mean_eval_mean_z"]].round(4).to_dict(orient="records")
    walk = {}
    for name in ("events", "placebo"):
        d = _read_csv(results / "eval" / f"walk_forward_{name}.csv")
        if d is not None:
            walk[name] = d[["split", "spearman", "auc_Z"]].round(4).to_dict(orient="records")

    def _ci(m):
        return (m.get("spearman_Z_vs_eval_mean_z") or {}).get("ci95")

    # ---- facts the Method page quotes (all computed here, nothing hard-coded on the page)
    rep = {}
    rp = results / "validation_report.json"
    if rp.exists():
        rep = json.loads(rp.read_text())

    def _sens(name):
        rows = [r for r in rep.get(f"sensitivity_{name}", []) if r.get("eval_mean_z_diff_vs_matched") is not None]
        lo = [(r.get("eval_mean_z_diff_ci95") or [None])[0] for r in rows]
        return {"settings": len(rows),
                "positive": int(sum(r["eval_mean_z_diff_vs_matched"] > 0 for r in rows)),
                "ci_above_zero": int(sum(x is not None and x > 0 for x in lo))}

    def _grp(name):
        r = rep.get(name, {})
        c, e = r.get("calibration", {}), r.get("evaluation", {})
        return {"flagged": c.get("flagged"), "eligible": c.get("eligible_wallets"), "flag_rate": c.get("flag_rate"),
                "matched_diff": e.get("mean_z_diff_vs_matched"), "matched_ci": e.get("mean_z_diff_vs_matched_ci95")}

    def _dec(name):
        d = deciles.get(name)
        return {"top": d[-1]["mean_eval_mean_z"], "bottom": d[0]["mean_eval_mean_z"]} if d else None

    syn = rep.get("synthetic", {})
    bh = lambda rows: [x for x in rows if x.get("method") == "BH"]
    sim = None
    if syn:
        n, p_ = bh(syn.get("noise_only", [])), bh(syn.get("planted", []))
        sim = {"noise_flag_rate": float(np.mean([x["flag_rate"] for x in n])) if n else None,
               "planted_recall": float(np.mean([x["recall"] for x in p_])) if p_ else None,
               "planted_precision": float(np.mean([x["precision"] or 0 for x in p_])) if p_ else None}
    validation = {"events": _grp("events"), "placebo": _grp("placebo"),
                  "sensitivity": {"events": _sens("events"), "placebo": _sens("placebo")},
                  "deciles": {"events": _dec("events"), "placebo": _dec("placebo")},
                  "simulation": sim, "old_test_placebo_flag_rate": 0.045,
                  "min_events": (rep.get("args") or {}).get("min_events", 10),
                  "fdr_q": (rep.get("args") or {}).get("q", 0.10)}

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "flagged_total": flagged_n, "eligible_total": int(len(scores)),
            "flag_cutoff": cutoff,
            "split": ev_m.get("split"),
            "events_spearman": (ev_m.get("spearman_Z_vs_eval_mean_z") or {}).get("rho"),
            "events_spearman_ci": _ci(ev_m),
            "placebo_spearman": (pl_m.get("spearman_Z_vs_eval_mean_z") or {}).get("rho"),
            "placebo_spearman_ci": _ci(pl_m),
            "events_auc": (ev_m.get("auc_beat_prices") or {}).get("Z"),
            "placebo_auc": (pl_m.get("auc_beat_prices") or {}).get("Z"),
            "events_precision_top100": (ev_m.get("precision_at_k_beat_prices") or {}).get("100"),
            "base_rate": ev_m.get("base_rate_beat_prices"),
        },
        "histogram": _histogram(scores["Z_adj"].to_numpy()),
        "deciles": deciles, "walk_forward": walk, "validation": validation, "wallets": wallets,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.json").write_text(json.dumps(payload, default=_j))
    print(f"wrote {out / 'data.json'}  ({len(wallets)} wallets, {flagged_n} flagged)")


def export_findings(results: Path, site_root: Path):
    """Copy the writeup figures into the site (web-sized PNG + original PDF) and write
    site/data/findings.json with captions, tables and key numbers. Needs make_report_figures.py to have run."""
    import re
    import shutil
    from PIL import Image
    rep = results / "report"
    if not (rep / "captions.md").exists():
        print("findings: results_v2/report not found (run make_report_figures.py); skipped")
        return
    fig_dir = site_root / "assets" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for line in (rep / "captions.md").read_text().splitlines():
        m = re.match(r"\*\*(Figure (\d+)\. [^*]+?)\*\*\s+(.*)", line)
        if not m:
            continue
        num = int(m.group(2))
        src = next(iter(sorted(rep.glob(f"fig{num}_*.png"))), None)
        if src is None:
            continue
        im = Image.open(src)
        if im.width > 1800:
            im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
        im.save(fig_dir / src.name, optimize=True)
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, fig_dir / pdf.name)
        figures.append({"id": src.stem, "title": m.group(1).rstrip("."), "caption": m.group(3),
                        "png": f"assets/figures/{src.name}", "pdf": f"assets/figures/{src.stem}.pdf"})
    tables = {}
    for key, fn in (("data", "table1_data"), ("main", "table2_main_results"), ("sensitivity", "table3_sensitivity"),
                    ("simulation", "table4_simulation"), ("walk_forward", "table5_walk_forward")):
        d = _read_csv(rep / f"{fn}.csv")
        if d is not None:
            tables[key] = json.loads(d.round(4).to_json(orient="records"))
    sections, cur = [], None
    kn = rep / "key_numbers.md"
    if kn.exists():
        for line in kn.read_text().splitlines():
            if line.startswith("## "):
                cur = {"title": line[3:].strip(), "bullets": []}
                sections.append(cur)
            elif line.startswith("- ") and cur is not None:
                cur["bullets"].append(line[2:].strip())
    (site_root / "data").mkdir(parents=True, exist_ok=True)
    (site_root / "data" / "findings.json").write_text(json.dumps({"figures": figures, "tables": tables,
                                                                   "key_numbers": sections}, default=_j))
    print(f"wrote findings.json ({len(figures)} figures, {len(tables)} tables)")


def export_alerts(results: Path, data: Path, site_root: Path, max_rows: int = 400):
    """site/data/alerts.json for the Alerts page: backtest table, lead-time stats, replayed alerts with outcomes,
    figures copied into the site. Needs evaluate_alerts.py and make_alert_figures.py to have run."""
    import shutil
    from PIL import Image
    a = results / "alerts"
    if not (a / "backtest_table.csv").exists() or not (a / "replay_alerts.parquet").exists():
        print("alerts: results_v2/alerts not found (run build_alert_candidates.py, evaluate_alerts.py); skipped")
        return
    names = pd.read_parquet(data / "wallet_codes.parquet")
    names = names.set_index("code")["wallet"] if "code" in names.columns else names["wallet"]
    bt = pd.read_csv(a / "backtest_table.csv")
    main_cut = "2026-06-01" if "2026-06-01" in set(bt["cutoff"]) else sorted(set(bt["cutoff"]))[0]
    cols = ["strategy", "delay", "slippage", "alerts", "events", "wallets", "roi", "roi_lo", "roi_hi", "win_rate",
            "random_mean", "random_p05", "random_p95", "mean_price_paid", "top_wallet_alert_share", "top_profit_wallet_share",
            "roi_without_top_wallet", "roi_wo_lo", "roi_wo_hi"]
    main = bt[bt["cutoff"] == main_cut][cols]
    robust = bt[(bt["delay_s"] == 300) & (bt["slippage"] == 0)][["cutoff", "strategy", "alerts", "roi", "roi_lo", "roi_hi"]]

    def replay(fname):
        f = a / fname
        if not f.exists():
            return []
        r = pd.read_parquet(f).sort_values("ts", ascending=False).head(max_rows)
        out = []
        for x in r.itertuples():
            out.append({"time": pd.Timestamp(x.ts).strftime("%Y-%m-%d %H:%M"), "wallet": str(names.get(x.wallet, x.wallet)).lower(),
                        "market": x.market, "held": x.held_label, "entry_price": round(float(x.p_exec), 3), "usd": round(float(x.usdc), 0),
                        "score": round(float(x.score), 3), "past_score": None if pd.isna(x.Z_past) else round(float(x.Z_past), 2),
                        "past_events": int(x.n_past), "hours_before_resolution": None if pd.isna(x.hours_to_resolve) else round(float(x.hours_to_resolve), 1),
                        "won": bool(x.payout > 0.5), "return": round(float(x.ret), 3), "pnl": round(float(x.pnl), 2),
                        "link": f"https://polymarket.com/event/{x.event_slug}" if isinstance(x.event_slug, str) and x.event_slug else None})
        return out

    fig_dir = site_root / "assets" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    caps = json.loads((a / "captions_alerts.json").read_text()) if (a / "captions_alerts.json").exists() else {}
    figures = []
    for stem, cap in caps.items():
        src = a / "figures" / f"{stem}.png"
        if not src.exists():
            continue
        im = Image.open(src)
        if im.width > 1800:
            im = im.resize((1800, round(im.height * 1800 / im.width)), Image.LANCZOS)
        im.save(fig_dir / src.name, optimize=True)
        if (a / "figures" / f"{stem}.pdf").exists():
            shutil.copy2(a / "figures" / f"{stem}.pdf", fig_dir / f"{stem}.pdf")
        figures.append({"id": stem, "title": cap.split(". ")[0], "caption": cap.split(". ", 1)[1] if ". " in cap else cap,
                        "png": f"assets/figures/{stem}.png", "pdf": f"assets/figures/{stem}.pdf"})
    lt = _read_csv(a / "lead_time_stats.csv")
    info = json.loads((a / "run_info.json").read_text()) if (a / "run_info.json").exists() else {}
    live = json.loads((a / "live_model_info.json").read_text()) if (a / "live_model_info.json").exists() else {}
    payload = {"generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"), "cutoff": main_cut,
               "backtest": json.loads(main.round(4).to_json(orient="records")),
               "robust": json.loads(robust.round(4).to_json(orient="records")),
               "lead_time": [] if lt is None else json.loads(lt.round(3).reset_index().to_json(orient="records")),
               "replay_model": replay("replay_alerts.parquet"), "replay_rule": replay("replay_rule_alerts.parquet"),
               "figures": figures, "run_info": info.get(main_cut, {}), "live_model": live}
    (site_root / "data").mkdir(parents=True, exist_ok=True)
    (site_root / "data" / "alerts.json").write_text(json.dumps(payload, default=_j))
    print(f"wrote alerts.json ({len(payload['replay_model'])} model alerts, {len(payload['replay_rule'])} rule alerts, {len(figures)} figures)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events/live")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--top-n", type=int, default=150, help="flagged wallets + highest-scoring near-misses to show")
    ap.add_argument("--trades-per-wallet", type=int, default=40)
    ap.add_argument("--scores-file", default="wallet_scores_live.csv")
    a = ap.parse_args()
    build(Path(a.data), Path(a.results), Path(a.out), a.top_n, a.trades_per_wallet, a.scores_file)
    export_findings(Path(a.results), Path(a.out).parent)
    export_alerts(Path(a.results), Path(a.data), Path(a.out).parent)
