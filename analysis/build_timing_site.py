"""Builds web/data/timing_explorer.json for the website's Trader Timing page from results_v4/timing, the flagged-wallet
list and the fresh-window alerts. Run from the project root:
    python build_timing_site.py --data data_events_full/live --results results_v4 --alerts results_v3/alerts_fresh/primary_alerts_5min.parquet --out polyinsiders_repo/web/data/timing_explorer.json
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

HRES_EDGES = [0, 1, 6, 24, 72, 168, 720, 2160, 1e9]
HRES_LABELS = ["under 1 h", "1-6 h", "6-24 h", "1-3 days", "3-7 days", "7-30 days", "30-90 days", "90+ days"]


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def rows(df):
    return [{k: clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events_full/live"); ap.add_argument("--results", default="results_v4")
    ap.add_argument("--alerts", default="results_v3/alerts_fresh/primary_alerts_5min.parquet")
    ap.add_argument("--out", default="polyinsiders_repo/web/data/timing_explorer.json")
    ap.add_argument("--top-trades", type=int, default=8)
    a = ap.parse_args()
    res, data = Path(a.results), Path(a.data); T = res / "timing"
    summ = json.loads((T / "timing_summary.json").read_text()); info = json.loads((T / "timing_info.json").read_text())
    clocks = pd.read_csv(T / "timing_clock_shares.csv")
    out = {"split": info["split"], "markets": info["markets"], "markets_with_shift": info["markets_with_shift"],
           "markets_with_sharp_jump": info["markets_with_sharp_jump"], "groups": info["groups"], "summary": summ,
           "clock_shares": rows(clocks), "by_shift": rows(pd.read_csv(T / "timing_by_shift_bucket.csv")),
           "fomc": rows(pd.read_csv(T / "timing_fomc.csv")), "markouts": rows(pd.read_csv(T / "timing_markouts.csv"))}

    mk = pd.read_parquet(data / "markets_coded.parquet", columns=["condition_id", "question", "end_time"]).set_index("condition_id")
    wc = pd.read_parquet(data / "wallet_codes.parquet").set_index("code")["wallet"]
    wl = pd.read_csv(res / "likely_informed_wallets.csv"); wl = wl[wl["informed"]].sort_values("Z_adj", ascending=False)
    addr2code = pd.Series(wc.index.values, index=wc.values)
    wl["code"] = wl["wallet"].map(addr2code)

    t = pd.read_parquet(T / "timing_trades.parquet")
    f = t[(t.grp & 2) > 0].copy()
    f["hb"] = pd.cut(f["hours_to_resolve"], HRES_EDGES, right=False, labels=False)
    ok = f["entry_price"].between(0.02, 0.98)
    prof = {}
    for code, g in f.groupby("wallet"):
        u = g["usdc"].to_numpy(dtype=float); tot = u.sum()
        if tot <= 0:
            continue
        hb = np.array([u[g["hb"].to_numpy() == i].sum() for i in range(8)]) / tot
        life = np.histogram(g["frac_life"].to_numpy(), bins=np.linspace(0, 1, 11), weights=u)[0] / tot
        d = {"trades": int(len(g)), "dollars": round(float(tot)), "markets": int(g["condition_id"].nunique()),
             "by_time_to_resolution": [round(float(x), 4) for x in hb], "by_life": [round(float(x), 4) for x in life],
             "last24h": round(float(hb[:3].sum()), 4), "first10pct": round(float(life[0]), 4)}
        gg = g[ok.loc[g.index]]
        for h in ["m5m", "m1h", "m24h"]:
            x = gg.dropna(subset=[h]); uu = x["usdc"].to_numpy(dtype=float)
            d[h] = round(float((x[h] / x["entry_price"] * uu).sum() / uu.sum()), 4) if uu.sum() > 0 else None
        e = gg["edge"] / gg["entry_price"]; uu = gg["usdc"].to_numpy(dtype=float)
        d["at_resolution"] = round(float((e * uu).sum() / uu.sum()), 4) if uu.sum() > 0 else None
        big = g.nlargest(a.top_trades, "usdc")
        d["top_trades"] = [{"t": int(r.ts_sec), "market": str(mk["question"].get(int(r.condition_id), ""))[:110], "usd": round(float(r.usdc)),
                            "price": round(float(r.entry_price), 3), "hours_before": round(float(r.hours_to_resolve), 1),
                            "edge": round(float(r.edge), 3)} for r in big.itertuples()]
        prof[int(code)] = d
    ws = []
    for r in wl.itertuples():
        d = {"address": r.wallet, "events": int(r.n_events), "z": round(float(r.Z_adj), 2), "profit": round(float(r.profit)),
             "volume": round(float(r.volume)), "ppd": round(float(r.profit_per_dollar), 4), "rule_r0": None}
        p = prof.get(int(r.code)) if pd.notna(r.code) else None
        d["timing"] = p
        ws.append(d)
    out["wallets"] = ws
    out["wallets_with_timing"] = sum(1 for w in ws if w["timing"])

    al = pd.read_parquet(a.alerts).sort_values("ts", ascending=False)
    out["alerts"] = [{"t": int(r.ts.timestamp()), "wallet": wc.get(int(r.wallet), ""), "market": str(mk["question"].get(int(r.condition_id), ""))[:110],
                      "usd_basis": 100, "price": round(float(r.entry_price), 3), "z": round(float(r.Z_past), 1), "past_events": int(r.n_past),
                      "won": int(r.payout > 0.5), "ret": round(float(r.ret), 3), "resolved": int(pd.Timestamp(r.event_time).timestamp())}
                     for r in al.itertuples()]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {a.out}: {len(ws)} wallets ({out['wallets_with_timing']} with timing), {len(out['alerts'])} alerts, {Path(a.out).stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
