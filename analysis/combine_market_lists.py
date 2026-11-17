"""Merge the market lists saved by several download runs into one file, so the analysis can run
OFFLINE (for example on Adroit's compute nodes, which have no internet) over all the data at once.

    python combine_market_lists.py data_events/live data_events_2025/live data_events_fresh/live --out data_events_full_markets.parquet

Each run folder must contain markets.parquet (written by run_milestone1.py). Markets that appear in more than one
list are kept once. Nothing is deleted or changed in the run folders. Then analyse the combined set, reading the
trade files already in the shared cache:

    python run_milestone1.py --out data_events_full --trades-dir data/live/trades_raw \\
        --market-list data_events_full_markets.parquet --n-markets 100000 --end-min 2025-01-01 --end-max 2026-11-15 \\
        --audit-onchain 0 --stage analyze
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pm473 import collect


def combine(run_dirs, trades_dir=None) -> pd.DataFrame:
    parts = []
    for d in run_dirs:
        f = Path(d) / "markets.parquet"
        if not f.exists():
            raise SystemExit(f"{f} not found. Run the download stage for that folder first.")
        m = collect.load_market_list(f)
        m["source_run"] = str(d)
        parts.append(m)
        print(f"  {d}: {len(m):,} markets")
    df = pd.concat(parts, ignore_index=True).drop_duplicates("condition_id", keep="first")
    if trades_dir is not None:
        have = {p.stem for p in Path(trades_dir).glob("0x*.parquet")}
        df["trades_downloaded"] = df["condition_id"].isin(have)
        print(f"  trade files on disk for {int(df['trades_downloaded'].sum()):,} of {len(df):,} markets")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run folders such as data_events/live")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trades-dir", default="data/live/trades_raw", help="report how many markets have trade files")
    a = ap.parse_args()
    df = combine(a.run_dirs, a.trades_dir if Path(a.trades_dir).exists() else None)
    df.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(df):,} markets, resolved {df['end_time'].min()} to {df['end_time'].max()}")


if __name__ == "__main__":
    main()
