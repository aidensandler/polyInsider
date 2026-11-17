"""Per-trade return and PnL (for the dashboard and for reporting).

Every fill is re-expressed as a LONG position in one outcome token:
  BUY outcome o at price p      -> long token o at p
  SELL outcome o at price p     -> long token (1 - o) at 1 - p   (a sale is a bet on the other side)
In outcome-0 units that is: dir0 > 0 -> long outcome 0 at p0, else long outcome 1 at 1 - p0.

  trade_return = (payout - entry_price) / entry_price     (e.g. +1.5 = +150%)
  trade_pnl    = dollars made on the fill, after fees      (= dir0 * size * (Y - p0) - fee)

Return is a per-dollar measure and can be huge for longshots; PnL is dollars.
Classification uses neither directly (see skill_test.py); the dashboard shows both.
Unresolved markets get NaN (the trade is still open).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_trade_returns(fills: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    """Add entry_price, held_outcome, trade_return, trade_pnl to a copy of `fills`.
    fills needs: condition_id, dir0, size, p0 (+ fee_usdc). mk needs: condition_id, value_0."""
    f = fills.copy()
    y = f["condition_id"].map(mk.set_index("condition_id")["value_0"]).astype("float64")
    d = f["dir0"].to_numpy()
    p0 = f["p0"].to_numpy(dtype="float64")
    long0 = d > 0
    f["held_outcome"] = np.where(long0, 0, 1).astype("int8")
    f["entry_price"] = np.where(long0, p0, 1.0 - p0)
    payout = np.where(long0, y.to_numpy(), 1.0 - y.to_numpy())
    f["trade_return"] = (payout - f["entry_price"]) / f["entry_price"]
    fee = f["fee_usdc"].to_numpy(dtype="float64") if "fee_usdc" in f else 0.0
    f["trade_pnl"] = d * f["size"].to_numpy(dtype="float64") * (y.to_numpy() - p0) - fee
    f["resolved"] = y.notna().to_numpy()
    return f


def wallet_trade_summary(trades: pd.DataFrame) -> pd.DataFrame:
    """Per wallet: number of resolved trades, mean/median return per trade, win rate,
    total PnL and PnL per dollar traded."""
    t = trades[trades["resolved"]]
    g = t.groupby("wallet")
    out = pd.DataFrame({
        "resolved_trades": g.size(),
        "mean_trade_return": g["trade_return"].mean(),
        "median_trade_return": g["trade_return"].median(),
        "win_rate": g["trade_pnl"].apply(lambda s: float((s > 0).mean())),
        "total_pnl": g["trade_pnl"].sum(),
        "volume": g["usdc"].sum() if "usdc" in t else np.nan,
    })
    out["pnl_per_dollar"] = out["total_pnl"] / out["volume"].replace(0, np.nan)
    return out
