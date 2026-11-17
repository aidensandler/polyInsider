"""Step 2b: verify the Data API fill tape against Polygon itself.

Your proposal (following Dubach 2026) commits to on-chain records as the
source of truth. Pulling *every* fill from chain logs is expensive; instead,
audit a random sample of Data API rows against the OrderFilled events in their
transaction receipts and report the match rate. If it is ~100%, the Data API
tape is a faithful (and much cheaper) view of on-chain executions.

Two event layouts exist:
  V1 (before 2026-04-28): OrderFilled(bytes32 indexed orderHash, address indexed maker,
       address indexed taker, uint256 makerAssetId, uint256 takerAssetId,
       uint256 makerAmountFilled, uint256 takerAmountFilled, uint256 fee)
  V2 (CTF Exchange V2):   OrderFilled(bytes32 indexed orderHash, address indexed maker,
       address indexed taker, uint8 side, uint256 tokenId, uint256 makerAmountFilled,
       uint256 takerAmountFilled, uint256 fee, bytes32 builder, bytes32 metadata)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Crypto.Hash import keccak

from . import config

SIG_V1 = "OrderFilled(bytes32,address,address,uint256,uint256,uint256,uint256,uint256)"
SIG_V2 = "OrderFilled(bytes32,address,address,uint8,uint256,uint256,uint256,uint256,bytes32,bytes32)"


def topic0(sig: str) -> str:
    k = keccak.new(digest_bits=256)
    k.update(sig.encode())
    return "0x" + k.hexdigest()


T_V1, T_V2 = topic0(SIG_V1), topic0(SIG_V2)
SCALE = 10 ** config.TOKEN_DECIMALS


def _words(data_hex: str) -> list[int]:
    h = data_hex[2:] if data_hex.startswith("0x") else data_hex
    return [int(h[i:i + 64], 16) for i in range(0, len(h), 64)]


def _addr(topic: str) -> str:
    return "0x" + topic[-40:].lower()


def decode_order_filled(log: dict) -> dict | None:
    """Returns a decoded fill, or None for any log that isn't an OrderFilled
    from a known exchange (OrdersMatched, FeeCharged, transfers, ...)."""
    addr = log.get("address", "").lower()
    t = log.get("topics") or []
    if addr not in config.EXCHANGES.values() or not t or t[0] not in (T_V1, T_V2) or len(t) < 3:
        return None
    w = _words(log.get("data", "0x"))
    base = {"tx_hash": log["transactionHash"].lower(), "log_index": int(log["logIndex"], 16),
            "exchange": addr, "maker": _addr(t[2]),
            "taker": _addr(t[3]) if len(t) > 3 else None}
    if t[0] == T_V1 and len(w) >= 5:
        maker_asset, taker_asset, mamt, tamt, fee = w[:5]
        if maker_asset == 0:   # maker pays collateral -> maker BUYS taker_asset
            side, token, shares, usdc = "BUY", taker_asset, tamt, mamt
        else:
            side, token, shares, usdc = "SELL", maker_asset, mamt, tamt
        ver = "v1"
    elif t[0] == T_V2 and len(w) >= 5:
        s, token, mamt, tamt, fee = w[:5]
        if s == 0:
            side, shares, usdc = "BUY", tamt, mamt
        else:
            side, shares, usdc = "SELL", mamt, tamt
        ver = "v2"
    else:
        return None
    shares, usdc = shares / SCALE, usdc / SCALE
    return {**base, "version": ver, "side": side, "token_id": str(token), "shares": shares,
            "usdc": usdc, "price": usdc / shares if shares else np.nan, "fee": fee / SCALE}


class ReceiptSource:
    """Receipts via a Polygon JSON-RPC URL (preferred: set POLYGON_RPC_URL),
    or Etherscan API V2 (set ETHERSCAN_API_KEY). The Etherscan free tier has
    dropped some chains; if you see 'Free API access is not supported for this
    chain', switch to an RPC provider (Alchemy/Infura/QuickNode free tiers work)."""

    def __init__(self, rpc_url=None, etherscan_key=None, min_interval=0.25):
        self.rpc = rpc_url or os.getenv("POLYGON_RPC_URL")
        self.key = etherscan_key or os.getenv("ETHERSCAN_API_KEY")
        if not (self.rpc or self.key):
            raise RuntimeError("Set POLYGON_RPC_URL or ETHERSCAN_API_KEY")
        self.min_interval, self._last = min_interval, 0.0
        self.s = requests.Session()

    def receipt(self, tx: str) -> dict | None:
        time.sleep(max(0, self.min_interval - (time.time() - self._last)))
        self._last = time.time()
        if self.rpc:
            r = self.s.post(self.rpc, json={"jsonrpc": "2.0", "id": 1,
                                            "method": "eth_getTransactionReceipt", "params": [tx]}, timeout=30)
        else:
            r = self.s.get(config.ETHERSCAN_V2_URL, params={
                "chainid": config.POLYGON_CHAIN_ID, "module": "proxy",
                "action": "eth_getTransactionReceipt", "txhash": tx, "apikey": self.key}, timeout=30)
        body = r.json()
        if isinstance(body.get("result"), str):
            raise RuntimeError(f"receipt error: {body['result']}")
        return body.get("result")


def audit_sample(fills: pd.DataFrame, n: int = 200, seed: int = 473,
                 source: ReceiptSource | None = None, cache_dir: str | Path | None = None):
    """Returns (summary dict, per-row match table). Receipts are cached in
    cache_dir, so an interrupted audit resumes; a receipt that fails is
    counted and skipped rather than stopping the run."""
    source = source or ReceiptSource()
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    sample = fills.dropna(subset=["tx_hash"]).sample(min(n, len(fills)), random_state=seed)
    decoded, unknown_topics, errors = {}, {}, 0
    txs = sample["tx_hash"].str.lower().unique()
    for i, tx in enumerate(txs):
        cf = cache / f"{tx}.json" if cache else None
        try:
            if cf and cf.exists():
                rc = json.loads(cf.read_text())
            else:
                rc = source.receipt(tx)
                if cf and rc is not None:
                    cf.write_text(json.dumps(rc))
        except Exception as e:      # noqa: BLE001
            errors += 1
            print(f"  [audit] receipt {tx[:10]}… failed: {str(e)[:100]}")
            continue
        rows = []
        for lg in (rc or {}).get("logs", []):
            d = decode_order_filled(lg)
            if d:
                rows.append(d)
            elif lg.get("address", "").lower() in config.EXCHANGES.values() and lg.get("topics"):
                k = lg["topics"][0]
                unknown_topics[k] = unknown_topics.get(k, 0) + 1
        decoded[tx] = rows
        if (i + 1) % 50 == 0:
            print(f"  [audit] {i+1}/{len(txs)} receipts")

    out = []
    for _, r in sample.iterrows():
        tx = r["tx_hash"].lower()
        if tx not in decoded:
            continue
        cands = [d for d in decoded[tx]
                 if d["token_id"] == r["token_id"] and r["wallet"] in (d["maker"], d["taker"])]
        size_ok = [d for d in cands if abs(d["shares"] - r["size"]) <= 1e-4 * max(1, r["size"])]
        side_ok = [d for d in size_ok if d["side"] == r["side"] or d["taker"] == r["wallet"]]
        out.append({"tx_hash": r["tx_hash"], "wallet": r["wallet"], "ts": r["ts"],
                    "found_tx_fill": bool(cands), "size_match": bool(size_ok),
                    "side_match": bool(side_ok), "n_orderfilled_in_tx": len(decoded[tx])})
    t = pd.DataFrame(out)
    summary = {"rows_checked": len(t), "receipt_errors": errors,
               **({f"rate_{c}": float(t[c].mean()) for c in ("found_tx_fill", "size_match", "side_match")}
                  if len(t) else {}),
               "other_exchange_event_topics": unknown_topics,
               "note": ("A taker's own side can appear inverted in the maker's OrderFilled event; "
                        "side_match therefore accepts the taker address on either side.")}
    return summary, t
