"""Step 1 of the alert model: one row per trade with features known at the time of the trade.

    python build_alert_candidates.py                       # events sample (needs data_events/live and results_v2)
    python build_alert_candidates.py --stop-after 8        # only 8 chunks this run (re-run to continue)

Writes results_v2/alerts/candidate_parts/part_XXXX.parquet (resumable: finished parts are skipped) and
results_v2/alerts/candidates.parquet when every part exists. Safe to stop and restart.
Keeps trades of at least $5 by wallets with at least 3 resolved events before the trade.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pm473 import alerts as al
from pm473 import skill_test as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_events/live")
    ap.add_argument("--results", default="results_v2")
    ap.add_argument("--groups-per-chunk", type=int, default=60)
    ap.add_argument("--min-past", type=int, default=3)
    ap.add_argument("--min-usd", type=float, default=5.0)
    ap.add_argument("--stop-after", type=int, default=0, help="stop after this many NEW chunks (0 = no limit)")
    a = ap.parse_args()

    res, data = Path(a.results), Path(a.data)
    out = res / "alerts"; parts_dir = out / "candidate_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(res / "events_table_events.parquet")
    mk = pd.read_parquet(data / "markets_coded.parquet")
    ek, pt = st.event_keys(mk), st.period_times(mk)
    hist = al.wallet_history(ev)
    pf = pq.ParquetFile(data / "fills_clean.parquet")
    n_groups = pf.metadata.num_row_groups
    starts = list(range(0, n_groups, a.groups_per_chunk))
    done_now, t0 = 0, time.time()
    for k, i in enumerate(starts):
        part = parts_dir / f"part_{k:04d}.parquet"
        if part.exists():
            continue
        tbl = pf.read_row_groups(list(range(i, min(n_groups, i + a.groups_per_chunk))), columns=al.FILL_COLS)
        c = al.trade_features(tbl.to_pandas(), mk, ek, pt, hist, a.min_past, a.min_usd)
        c.to_parquet(part, index=False)
        done_now += 1
        print(f"chunk {k + 1}/{len(starts)}: {len(c):,} trades kept  ({time.time() - t0:.0f}s)", flush=True)
        if a.stop_after and done_now >= a.stop_after:
            break
    missing = [k for k in range(len(starts)) if not (parts_dir / f"part_{k:04d}.parquet").exists()]
    if missing:
        print(f"{len(missing)} chunks still to do; run again to continue.")
        return
    # stream the parts into one file (low memory: one part at a time)
    target = out / "candidates.parquet"
    writer, n_rows, wallets = None, 0, set()
    for k in range(len(starts)):
        tbl = pq.read_table(parts_dir / f"part_{k:04d}.parquet")
        if writer is None:
            writer = pq.ParquetWriter(target, tbl.schema)
        writer.write_table(tbl)
        n_rows += tbl.num_rows
        wallets.update(pd.unique(tbl.column("wallet").to_numpy()))
    writer.close()
    import shutil
    shutil.rmtree(parts_dir, ignore_errors=True)          # the parts are only needed to resume an interrupted run
    print(f"wrote {target}: {n_rows:,} trades, {len(wallets):,} wallets")


if __name__ == "__main__":
    main()
