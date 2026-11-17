"""Step 3: realized PnL per (wallet, market) at resolution.

Why this works from trades alone (no split/merge/redeem history needed):
  * A wallet that SELLs shares it never bought must have minted them via a
    split ($1 -> 1 share of each outcome). If its net position in an outcome
    is -k, we add k complete sets at $1 each. That reproduces its true
    holdings exactly.
  * Merges ($1 back for one of each outcome) and redemptions (payout at
    resolution) are both captured by valuing the end inventory at the
    resolution payout.
So for a binary market, PnL = cash from trades - split cost + payout(inventory).
Caveat: tokens moved between wallets by plain ERC-1155 transfers, and neg-risk
conversions across markets, are not seen. That is why neg-risk markets are
excluded by default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wallet_market_pnl(fills: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    f = fills
    is_buy = f["is_buy"].to_numpy(dtype=bool) if "is_buy" in f else (f["side"] == "BUY").to_numpy(dtype=bool)
    sgn = np.where(is_buy, 1.0, -1.0)
    tmp = pd.DataFrame({
        "wallet": f["wallet"], "condition_id": f["condition_id"], "o": f["outcome_index"],
        "shares": sgn * f["size"], "cash": -sgn * f["usdc"] - f["fee_usdc"],
        "buy_usdc": np.where(sgn > 0, f["usdc"], 0.0), "usdc": f["usdc"],
    })
    g = tmp.groupby(["wallet", "condition_id", "o"]).agg(shares=("shares", "sum")).unstack("o", fill_value=0.0)
    g.columns = [f"n{c}" for c in g.columns.get_level_values(1)]
    for c in ("n0", "n1"):
        if c not in g:
            g[c] = 0.0
    tot = tmp.groupby(["wallet", "condition_id"]).agg(
        cash=("cash", "sum"), buy_usdc=("buy_usdc", "sum"), volume=("usdc", "sum"),
        n_fills=("usdc", "size"))
    out = g.join(tot).reset_index()

    eps = 1e-9
    out["splits"] = np.maximum(0, np.maximum(-out["n0"], -out["n1"]))
    out.loc[out["splits"] < eps, "splits"] = 0.0
    out["n0"] += out["splits"]
    out["n1"] += out["splits"]
    out["cash"] -= out["splits"]

    m = markets.set_index("condition_id")
    v0 = out["condition_id"].map(m["value_0"])
    v1 = out["condition_id"].map(m["value_1"])
    out["payout"] = out["n0"] * v0 + out["n1"] * v1
    out["pnl"] = out["cash"] + out["payout"]
    # Capital at risk: what the wallet paid in (buys + split collateral).
    out["capital"] = out["buy_usdc"] + out["splits"]
    out["roi"] = out["pnl"] / out["capital"].where(out["capital"] > 0)
    out["resolve_time"] = out["condition_id"].map(m["resolve_time"])
    # Outcomes of one multi-outcome event (e.g. candidates in one race) share
    # an event key; standalone markets get their own. Prefixes keep the two
    # kinds of ID from ever colliding.
    own = "m:" + out["condition_id"].astype(str)
    if "event_id" in m:
        ev = out["condition_id"].map(m["event_id"])
        out["event_key"] = np.where(ev.notna(), "e:" + ev.astype(str), own)
    else:
        out["event_key"] = own
    return out
