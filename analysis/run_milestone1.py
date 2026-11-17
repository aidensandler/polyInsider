"""Milestone I: collect -> clean -> PnL -> features -> informed/uninformed labels -> baseline comparison.

Examples
--------
# offline sanity check with planted informed traders (no network needed)
python run_milestone1.py --synthetic

# real data: binary markets that resolved Jan-Sep 2026, label on Jan-May, evaluate on Jun-Sep
python run_milestone1.py --end-min 2026-01-01 --end-max 2026-09-30 --split-date 2026-06-01 \
    --min-volume 50000 --n-markets 400

# add an on-chain audit of 200 random fills (needs POLYGON_RPC_URL or ETHERSCAN_API_KEY)
python run_milestone1.py ... --audit-onchain 200
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pm473 import classify, clean, collect, features, pnl
from pm473.api import Client


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--synthetic-markets", type=int, default=160)
    ap.add_argument("--end-min", default="2026-01-01")
    ap.add_argument("--end-max", default="2026-09-30")
    ap.add_argument("--split-date", default="2026-06-01",
                    help="markets resolving before this date = formation, after = evaluation")
    ap.add_argument("--min-volume", type=float, default=50_000)
    ap.add_argument("--max-volume", type=float, default=None)
    ap.add_argument("--n-markets", type=int, default=400)
    ap.add_argument("--tags", nargs="*", default=collect.DEFAULT_TAGS,
                    help="Gamma topic tags to sample from; pass --tags with nothing after it for all markets")
    ap.add_argument("--exclude-neg-risk", action="store_true",
                    help="drop outcomes of multi-outcome events (included by default)")
    ap.add_argument("--min-life-days", type=float, default=2.0,
                    help="drop markets open for less than this many days")
    ap.add_argument("--exclude-slug-pattern", default=collect.RECURRING_CRYPTO,
                    help='regex of market slugs to drop (default: recurring crypto). "" keeps all')
    ap.add_argument("--include-slug-pattern", default="",
                    help="regex: keep ONLY markets whose slug matches (e.g. for the crypto placebo)")
    ap.add_argument("--trades-dir", default=None,
                    help="where per-market trade files live; share it between runs to reuse downloads")
    ap.add_argument("--gamma-cache", nargs="*", default=None,
                    help="select from Gamma pages saved by an earlier run (e.g. data/live/raw) "
                         "instead of listing markets through the API")
    ap.add_argument("--market-list", default=None,
                    help="select from a markets.parquet saved by an earlier run (e.g. data/live/markets.parquet)")
    ap.add_argument("--prefer-downloaded", action="store_true",
                    help="when sampling, take already-downloaded eligible markets first")
    ap.add_argument("--min-markets", type=int, default=5, help="min formation markets to be labelled")
    ap.add_argument("--fdr", type=float, default=0.10)
    ap.add_argument("--audit-onchain", type=int, default=0)
    ap.add_argument("--out", default="data")
    ap.add_argument("--stage", choices=["all", "download", "analyze"], default="all",
                    help="download = fetch markets+trades only (light on memory); "
                         "analyze = process what's on disk, no trade downloads")
    a = ap.parse_args()

    out = Path(a.out) / ("synthetic" if a.synthetic else "live")
    out.mkdir(parents=True, exist_ok=True)
    report = {"args": vars(a)}

    # ---------------------------------------------------------- 1. collect
    print("1/6 selecting markets + downloading trades")
    if a.synthetic:
        from pm473 import synthetic
        gamma_raw, raw_trades, truth = synthetic.generate(n_markets=a.synthetic_markets)
        markets, sel_log = collect.filter_markets(gamma_raw)
    else:
        client = Client(raw_dir=out / "raw")
        tdir = Path(a.trades_dir) if a.trades_dir else out / "trades_raw"
        sel_key = {k: getattr(a, k) for k in ("end_min", "end_max", "min_volume", "max_volume",
                                               "n_markets", "exclude_neg_risk", "tags", "min_life_days",
                                               "exclude_slug_pattern", "include_slug_pattern",
                                               "prefer_downloaded", "gamma_cache", "market_list")}
        mf, sf = out / "markets.parquet", out / "selection.json"
        cached = json.loads(sf.read_text()) if sf.exists() else {}
        if mf.exists() and cached.get("args") == sel_key:
            markets, sel_log = pd.read_parquet(mf), cached["log"]
            print(f"   reusing saved market list ({len(markets)} markets; delete {mf.name} to reselect)")
        else:
            prefer = ({f.stem for f in tdir.glob("0x*.parquet")} if a.prefer_downloaded and tdir.exists()
                      else None)
            if a.market_list:
                saved = collect.load_market_list(a.market_list)
                print(f"   loaded {len(saved):,} markets from {a.market_list}")
                markets, sel_log = collect.filter_market_table(
                    saved, include_neg_risk=not a.exclude_neg_risk, n_markets=a.n_markets,
                    end_min=a.end_min, end_max=a.end_max, min_volume=a.min_volume,
                    max_volume=a.max_volume, min_life_days=a.min_life_days,
                    exclude_slug_pattern=a.exclude_slug_pattern,
                    include_slug_pattern=a.include_slug_pattern, prefer_ids=prefer)
            elif a.gamma_cache:
                raw = collect.load_cached_gamma(a.gamma_cache)
                print(f"   loaded {len(raw):,} markets from saved Gamma pages")
                markets, sel_log = collect.filter_markets(
                    raw, include_neg_risk=not a.exclude_neg_risk, n_markets=a.n_markets,
                    end_min=a.end_min, end_max=a.end_max, min_volume=a.min_volume,
                    max_volume=a.max_volume, min_life_days=a.min_life_days,
                    exclude_slug_pattern=a.exclude_slug_pattern,
                    include_slug_pattern=a.include_slug_pattern, prefer_ids=prefer)
            else:
                markets, sel_log = collect.select_markets(
                    client, a.end_min, a.end_max, a.min_volume, a.max_volume,
                    include_neg_risk=not a.exclude_neg_risk, n_markets=a.n_markets, tags=a.tags or None,
                    min_life_days=a.min_life_days, exclude_slug_pattern=a.exclude_slug_pattern,
                    include_slug_pattern=a.include_slug_pattern, prefer_ids=prefer)
            if len(markets) == 0:
                raise SystemExit("No markets were selected, so nothing was saved or downloaded. Check the tag names "
                                 "(pass them as separate words, not one quoted string), the --end-min/--end-max dates "
                                 "and --min-volume. Selection log: " + json.dumps(sel_log, default=_j))
            markets.to_parquet(mf, index=False)
            sf.write_text(json.dumps({"args": sel_key, "log": sel_log}, default=_j))
        print("   selection:", json.dumps(sel_log, default=_j))
        if len(markets) < a.n_markets:
            print(f"   note: only {len(markets)} eligible markets (asked for {a.n_markets}); "
                  "widen --end-min/--end-max or lower --min-volume for more")
        if a.stage == "analyze":
            files = [tdir / f"{c}.parquet" for c in markets["condition_id"] if (tdir / f"{c}.parquet").exists()]
            failed = [c for c in markets["condition_id"] if not (tdir / f"{c}.parquet").exists()]
            print(f"   analyze stage: {len(files)} downloaded markets ({len(failed)} not downloaded, skipped)")
        else:
            files, failed = collect.download_trades(client, markets, tdir)
        if failed:
            print(f"   {len(failed)} markets failed to download; re-run the same command to retry them")
            report["download_failed"] = failed
        if a.stage == "download":
            print(f"\ndownload stage done: {len(files)} markets on disk. "
                  "Run again with --stage analyze when you're ready to process.")
            return
    report["selection"] = sel_log

    # ------------------------------------------------------------ 2. clean
    print(f"2/6 cleaning   (peak memory so far: {_peak_gb():.1f} GB)")
    # IDs are stored as int codes from here on (wallet, condition_id, tx_hash);
    # wallet_codes.parquet and markets_coded.parquet translate them back.
    if a.synthetic:
        fills, clean_log = clean.normalize(raw_trades, markets)
        audit_rows = fills.head(0)
        del raw_trades
        fills = clean.slim(fills)
        fills, mk, wallet_names, cond_names = clean.encode_ids(fills, markets)
        fills.to_parquet(out / "fills_clean.parquet", index=False)
        markets.to_parquet(out / "markets.parquet", index=False)
    else:
        fills, clean_log, audit_rows, wallet_names, cond_names = clean.load_and_normalize(
            files, markets, audit_n=a.audit_onchain, out_path=out / "fills_clean.parquet")
        mk = clean.coded_markets(markets, cond_names)
    mk.to_parquet(out / "markets_coded.parquet", index=False)
    pd.DataFrame({"code": np.arange(len(wallet_names)), "wallet": wallet_names.astype(str)}) \
        .to_parquet(out / "wallet_codes.parquet", index=False)
    print(f"   {clean_log['clean_rows']:,} clean fills, "
          f"{fills.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory")
    report["cleaning"] = clean_log
    if a.audit_onchain and not a.synthetic:
        from pm473 import onchain
        print(f"   on-chain audit of {a.audit_onchain} random fills")
        try:
            summ, detail = onchain.audit_sample(audit_rows, n=len(audit_rows), cache_dir=out / "receipts")
            detail.to_csv(out / "onchain_audit.csv", index=False)
            report["onchain_audit"] = summ
            print("   audit:", json.dumps(summ, default=_j))
        except Exception as e:      # noqa: BLE001 -- the audit is a check, not a dependency
            report["onchain_audit"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"   audit failed ({e}); continuing without it. Re-run to retry (receipts are cached).")

    # -------------------------------------------------------------- 3. PnL
    print("3/6 PnL" + f"   (peak memory so far {_peak_gb():.2f} GB)")
    wm = pnl.wallet_market_pnl(fills, mk)
    _decoded(wm, wallet_names, cond_names).to_parquet(out / "wallet_market_pnl.parquet", index=False)
    report["pnl_sanity"] = {
        # Sum of all wallets' PnL should be ~ -fees if both sides of every fill
        # are in the tape. A large gap means counterparties are missing.
        "sum_pnl_all_wallets": float(wm["pnl"].sum()),
        "sum_volume": float(wm["volume"].sum()),
        "share_wallet_markets_with_splits": float((wm["splits"] > 0).mean()),
    }

    # --------------------------------------------------------- 4. features
    print("4/6 features" + f"   (peak memory so far {_peak_gb():.2f} GB)")
    gc.collect()
    fills = features.add_markouts(fills, mk)
    # Only formation-period fills are needed from here on; keep that half and
    # release the full table (evaluation uses the much smaller PnL table).
    split = pd.Timestamp(a.split_date, tz="UTC")
    # Whole events go to one period, by when their LAST outcome resolves, so a
    # single bet never straddles formation and evaluation.
    ev_id = mk["event_id"].astype("string") if "event_id" in mk else pd.Series(pd.NA, index=mk.index)
    ev_id = ev_id.fillna("m:" + mk["condition_id"].astype(str))
    period_time = mk.groupby(ev_id.to_numpy())["resolve_time"].transform("max")
    period_time = pd.Series(period_time.to_numpy(), index=mk["condition_id"].to_numpy())
    moved = int(((mk["resolve_time"] < split) & (period_time.to_numpy() >= split)).sum())
    report["split_events_moved_to_evaluation"] = moved
    in_form = (fills["condition_id"].map(period_time) < split).to_numpy()
    report["split"] = {"formation_markets": int(fills.loc[in_form, "condition_id"].nunique()),
                       "evaluation_markets": int(fills.loc[~in_form, "condition_id"].nunique())}
    f_form = fills[in_form]
    del fills, in_form
    gc.collect()
    wm_period = wm["condition_id"].map(period_time)
    p_form = wm[(wm_period < split).to_numpy()]
    p_eval = wm[(wm_period >= split).to_numpy()]
    wf = features.wallet_features(f_form)

    # -------------------------------------------------------- 5. classify
    print("5/6 labelling + models" + f"   (peak memory so far {_peak_gb():.2f} GB)")
    panel = classify.build_panel(f_form, p_form, p_eval, wf, a.min_markets, a.fdr)
    panel.index = pd.Index(wallet_names[panel.index.to_numpy()].astype(str), name="wallet")
    panel.to_csv(out / "wallet_panel.csv")
    report["labels"] = {"eligible_wallets": len(panel), "informed": int(panel["informed"].sum()),
                        "eligible_with_eval_trades": int(panel["eval_roi"].notna().sum())}
    report["baseline_comparison"] = classify.compare_to_baseline(panel)
    report["predictive_models"] = classify.predictive_models(panel)

    if a.synthetic:   # how well did we recover the planted traders?
        panel["truth"] = truth.reindex(panel.index)
        ct = pd.crosstab(panel["truth"], panel["informed"])
        flagged = panel[panel["informed"]]
        report["synthetic_recovery"] = {
            "confusion": ct.to_dict(),
            "precision": float((flagged["truth"] == "informed").mean()) if len(flagged) else None,
            "recall": float(flagged["truth"].eq("informed").sum()
                            / max(1, (panel["truth"] == "informed").sum())),
            "noise_false_positives": int((flagged["truth"] == "noise").sum()),
        }

    # ---------------------------------------------------------- 6. report
    print("6/6 report" + f"   (peak memory so far {_peak_gb():.2f} GB)")
    _figure(panel, out / "eval_roi_informed_vs_rest.png")
    (out / "milestone1_report.json").write_text(json.dumps(report, indent=2, default=_j))
    print(json.dumps(report, indent=2, default=_j))
    print(f"\noutputs in {out.resolve()}   (peak memory: {_peak_gb():.1f} GB)")


def _decoded(df, wallet_names, cond_names):
    """Copy of a table with int-coded wallet/condition_id turned back into IDs."""
    d = df.copy()
    codes = d["condition_id"].to_numpy()
    d["wallet"] = wallet_names[d["wallet"].to_numpy()].astype(str)
    d["condition_id"] = cond_names[codes].astype(str)
    if "event_key" in d:   # standalone markets: "m:<code>" -> "m:<condition id>"
        own = d["event_key"].astype(str).str.startswith("m:").to_numpy()
        ek = d["event_key"].astype(str).to_numpy(dtype=object)
        ek[own] = "m:" + d["condition_id"].to_numpy(dtype=object)[own]
        d["event_key"] = ek
    return d


def _peak_gb():
    import resource, sys
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6   # macOS reports bytes, Linux KB


def _figure(panel, path):
    p = panel.dropna(subset=["eval_roi"])
    if p.empty:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lo, hi = p["eval_roi"].quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 50)
    for lab, g in p.groupby("informed"):
        ax[0].hist(g["eval_roi"].clip(lo, hi), bins=bins, alpha=0.6, density=True,
                   label="informed" if lab else "uninformed")
    ax[0].set(title="Evaluation-period ROI (out of sample)", xlabel="pooled ROI", ylabel="density")
    ax[0].legend()
    ax[1].scatter(p["skill_t"], p["eval_roi"].clip(lo, hi), s=6, alpha=0.4,
                  c=np.where(p["informed"], "C1", "C0"))
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set(title="Formation skill t-stat vs evaluation ROI", xlabel="formation t-stat",
              ylabel="evaluation ROI")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def _j(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, tuple): return list(o)
    return str(o)


if __name__ == "__main__":
    main()
