"""Step 2: normalize raw Data API v2 rows into one tidy fill table.

Design rule: every filter is logged with a row count, and nothing is dropped
on the basis of outcomes or profitability (that would bias the PnL results).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# v2 is snake_case; keep camelCase fallbacks in case you load older v1 dumps.
COLUMN_ALIASES = {
    "wallet": ["proxy_wallet", "proxyWallet", "user", "wallet"],
    "condition_id": ["condition_id", "conditionId", "_condition_query"],
    "token_id": ["token_id", "asset", "asset_id"],
    "side": ["side"],
    "price": ["price"],
    "size": ["size", "shares"],
    "timestamp": ["timestamp", "match_time", "time"],
    "tx_hash": ["transaction_hash", "transactionHash", "tx_hash"],
    "fee_usdc": ["fee_usdc", "fee", "fees"],
    "role": ["role", "liquidity", "trader_side", "is_taker"],
    "row_id": ["id", "trade_id", "log_index", "logIndex"],
}


def _pick(raw: pd.DataFrame, names: list[str]):
    for n in names:
        if n in raw.columns:
            return raw[n]
    return None


STR = "string[pyarrow]"   # ~3-5x smaller than Python object strings on pandas 2.x


def load_and_normalize(files, markets: pd.DataFrame, audit_n: int = 0, seed: int = 473,
                       out_path=None):
    """Clean one market file at a time, appending each cleaned piece to a
    single parquet file (`out_path`), then read that file back once.

    Peak memory is about one copy of the cleaned table: raw feeds are never
    all in memory, and the read-back frees Arrow buffers as it converts.

    Returns (fills, log, audit_rows). `audit_rows` is a uniform random sample
    of `audit_n` fills with the full columns the on-chain audit needs; those
    columns (token_id, side text) are dropped from the main table."""
    import tempfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    wanted = {n for names in COLUMN_ALIASES.values() for n in names}
    rng = np.random.default_rng(seed)
    out_path = Path(out_path) if out_path else Path(tempfile.mkstemp(suffix=".parquet")[1])
    tmp = out_path.with_suffix(".partial")
    writer, schema, log = None, None, {}
    wallet_code, cond_code = {}, {}                     # running ID -> int code
    keep_keys, keep_rows = np.array([]), None          # streaming uniform sample
    for i, f in enumerate(files):
        cols = [c for c in pq.read_schema(f).names if c in wanted]
        raw = pd.read_parquet(f, columns=cols)
        if raw.empty:
            continue
        part, plog = normalize(raw, markets, sort=True)
        del raw
        for k, v in plog.items():
            if k not in ("wallets", "markets"):
                log[k] = log.get(k, 0) + v
        if not len(part):
            continue
        if audit_n:
            keys = rng.random(len(part))
            cand_keys = np.concatenate([keep_keys, keys])
            cand_rows = part if keep_rows is None else pd.concat([keep_rows, part], ignore_index=True)
            top = np.argsort(cand_keys)[:audit_n]
            keep_keys, keep_rows = cand_keys[top], cand_rows.iloc[top].reset_index(drop=True)
            del cand_rows
        part = slim(part)
        part = _encode_part(part, wallet_code, cond_code)
        if schema is None:
            table = pa.Table.from_pandas(part, preserve_index=False)
            schema = table.schema
            writer = pq.ParquetWriter(tmp, schema)
        else:
            table = pa.Table.from_pandas(part[schema.names], schema=schema, preserve_index=False)
        writer.write_table(table)
        del part, table
        if (i + 1) % 100 == 0:
            print(f"   cleaned {i+1}/{len(files)} market files")
    if writer is None:
        return pd.DataFrame(), log, keep_rows, np.array([], dtype=object), np.array([], dtype=object)
    writer.close()
    tmp.replace(out_path)
    df = read_fills(out_path)
    log["clean_rows"] = len(df)
    log["wallets"] = int(df["wallet"].nunique())
    log["markets"] = int(df["condition_id"].nunique())
    wallet_names = np.empty(len(wallet_code), dtype=object)
    wallet_names[list(wallet_code.values())] = list(wallet_code.keys())
    cond_names = np.empty(len(cond_code), dtype=object)
    cond_names[list(cond_code.values())] = list(cond_code.keys())
    return df, log, keep_rows, wallet_names, cond_names


def _codes(values: pd.Series, table: dict) -> np.ndarray:
    """Map strings to stable int codes, adding unseen ones to `table`."""
    uniq = pd.unique(values.astype(str))
    for u in uniq:
        if u not in table:
            table[u] = len(table)
    return values.astype(str).map(table).to_numpy(dtype="int32")


def tx_hash_code(tx: pd.Series) -> np.ndarray:
    """64-bit hash of transaction hashes (used only to group fills of one
    order within a wallet and market; collisions are negligible)."""
    return pd.util.hash_array(tx.astype(str).to_numpy(dtype=object)).view("int64")


def _encode_part(part, wallet_code, cond_code):
    part = part.copy()
    part["wallet"] = _codes(part["wallet"], wallet_code)
    part["condition_id"] = _codes(part["condition_id"], cond_code)
    part["tx_hash"] = tx_hash_code(part["tx_hash"])
    return part


def coded_markets(markets: pd.DataFrame, cond_names) -> pd.DataFrame:
    """Markets table keyed by the same int codes as the encoded fills."""
    cmap = pd.Series(np.arange(len(cond_names), dtype="int32"), index=pd.Index(cond_names).astype(str))
    mk = markets[markets["condition_id"].astype(str).isin(cmap.index)].copy()
    mk["condition_id_str"] = mk["condition_id"].astype(str)
    mk["condition_id"] = mk["condition_id_str"].map(cmap).astype("int32")
    return mk


def read_fills(path) -> pd.DataFrame:
    """Low-memory read: strings stay Arrow-backed, buffers freed as converted."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    return t.to_pandas(split_blocks=True, self_destruct=True,
                       types_mapper={pa.string(): pd.StringDtype("pyarrow"),
                                     pa.large_string(): pd.StringDtype("pyarrow")}.get)


def slim(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns the analysis doesn't need after cleaning and shrink dtypes."""
    df = df.drop(columns=[c for c in ("token_id", "side", "start_time") if c in df])
    for c in ("p0", "hours_to_resolve", "frac_life"):
        if c in df:
            df[c] = df[c].astype("float32")
    return df


def normalize(raw: pd.DataFrame, markets: pd.DataFrame, sort: bool = True) -> tuple[pd.DataFrame, dict]:
    log = {"raw_rows": len(raw)}
    df = pd.DataFrame({k: _pick(raw, v) for k, v in COLUMN_ALIASES.items()})
    missing = [c for c in ("wallet", "condition_id", "token_id", "side", "price", "size", "timestamp")
               if df[c].isna().all()]
    if missing:
        raise ValueError(f"Required fields not found: {missing}. Raw columns: {list(raw.columns)} "
                         "-- update COLUMN_ALIASES.")

    df["wallet"] = df["wallet"].astype(STR).str.lower()
    df["condition_id"] = df["condition_id"].astype(STR).str.lower()
    df["token_id"] = df["token_id"].astype(STR)
    df["side"] = df["side"].astype(STR).str.upper()
    df["tx_hash"] = df["tx_hash"].astype(STR)
    for c in ("price", "size", "fee_usdc"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.notna().any() and ts.median() > 1e11:        # milliseconds
        ts = ts / 1000
    df["ts"] = pd.to_datetime(ts, unit="s", utc=True).astype("datetime64[ns, UTC]")
    df["fee_usdc"] = df["fee_usdc"].fillna(0.0)

    def drop(name, mask):
        nonlocal df
        log[name] = int((~mask).sum())
        df = df[mask]

    drop("dropped_bad_fields", df["price"].between(0, 1, inclusive="neither")
         & (df["size"] > 0) & df["side"].isin(["BUY", "SELL"]) & df["ts"].notna())
    drop("dropped_infrastructure_wallets", ~df["wallet"].isin(config.NON_TRADER_ADDRESSES))

    # Identical rows are KEPT. An on-chain check of 50 such groups found every
    # copy was a separate fill (one order matching several resting orders at
    # the same price and size); the feed has no row id to tell them apart.
    # Only drop when the feed supplies a row id and that id repeats.
    key = ["tx_hash", "wallet", "token_id", "side", "price", "size", "ts"]
    log["identical_rows_kept"] = int(df.duplicated(key).sum())
    if df["row_id"].notna().any():
        before = len(df)
        df = df.drop_duplicates(key + ["row_id"])
        log["dropped_repeated_row_id"] = before - len(df)

    # attach market info + map token -> outcome index
    m = markets.set_index("condition_id")
    df = df[df["condition_id"].isin(m.index)]
    tok0 = df["condition_id"].map(m["token_0"].astype(str)).astype(STR)
    tok1 = df["condition_id"].map(m["token_1"].astype(str)).astype(STR)
    is0 = (df["token_id"] == tok0).fillna(False).to_numpy(dtype=bool)
    is1 = (df["token_id"] == tok1).fillna(False).to_numpy(dtype=bool)
    df["outcome_index"] = np.select([is0, is1], [0, 1], -1).astype("int8")
    drop("dropped_token_not_in_market", df["outcome_index"] >= 0)

    df["resolve_time"] = df["condition_id"].map(m["resolve_time"])
    df["start_time"] = df["condition_id"].map(m["start_time"])
    drop("dropped_after_resolution", df["ts"] <= df["resolve_time"])

    # derived columns
    df["usdc"] = df["price"] * df["size"]
    # Express every fill in "outcome-0 probability" units so YES and NO tapes merge
    df["p0"] = np.where(df["outcome_index"] == 0, df["price"], 1 - df["price"])
    # +1 = fill increases exposure to outcome 0 (BUY o0 or SELL o1)
    buy = (df["side"] == "BUY").fillna(False).to_numpy(dtype=bool)
    df["is_buy"] = buy
    df["dir0"] = np.where((df["outcome_index"] == 0) == buy, 1, -1)
    df["hours_to_resolve"] = (df["resolve_time"] - df["ts"]).dt.total_seconds() / 3600
    if "info_time" in m:
        # hours until the deciding information became public (negative = after)
        it = df["condition_id"].map(m["info_time"])
        df["hours_to_info"] = ((it - df["ts"]).dt.total_seconds() / 3600).astype("float32")
    life = (df["resolve_time"] - df["start_time"]).dt.total_seconds()
    df["frac_life"] = ((df["ts"] - df["start_time"]).dt.total_seconds() / life).clip(0, 1)
    df["post_v2_exchange"] = df["ts"] >= config.CLOB_V2_CUTOVER

    # drop helper columns that came back empty (e.g. no maker/taker field in the feed)
    df = df.drop(columns=[c for c in ("role", "row_id") if df[c].isna().all()])
    df["dir0"] = df["dir0"].astype("int8")
    if sort:
        df = df.sort_values(["condition_id", "ts"]).reset_index(drop=True)
    log["clean_rows"] = len(df)
    log["wallets"] = int(df["wallet"].nunique())
    log["markets"] = int(df["condition_id"].nunique())
    return df, log


def encode_ids(fills: pd.DataFrame, markets: pd.DataFrame):
    """In-memory version of the streaming encoding (used for synthetic data).
    Returns (fills, markets_coded, wallet_names, condition_names)."""
    w_codes, w_names = pd.factorize(fills["wallet"])
    c_codes, c_names = pd.factorize(fills["condition_id"])
    fills["wallet"] = w_codes.astype("int32")
    fills["condition_id"] = c_codes.astype("int32")
    if "tx_hash" in fills:
        fills["tx_hash"] = tx_hash_code(fills["tx_hash"])
    c_names = np.asarray(c_names, dtype=object)
    return fills, coded_markets(markets, c_names), np.asarray(w_names, dtype=object), c_names
