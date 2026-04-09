"""Trade execution engine with slippage modelling and portfolio tracking."""
import numpy as np
from typing import Any, Dict, Optional


class TradeExecutor:
    """Executes trades, applies slippage, and tracks portfolio state.

    Maintains internal portfolio state (positions, cash, PnL history) so that
    callers can execute trades sequentially and query statistics at any time.

    Slippage is modelled as a linear function of trade size relative to market
    liquidity. Larger trades consume more liquidity and move the market price
    against the trader - a key real-world constraint on Polymarket.

    Args:
        initial_cash: Starting cash balance
    """

    def __init__(self, initial_cash: float = 10_000.0):
        self._initial_cash = initial_cash
        self.reset()

    def reset(self) -> None:
        """Clear all portfolio state and return to initial conditions."""
        self._cash = self._initial_cash
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._trade_history: list = []
        self._pnl_history: list = []

    def compute_slippage(self, size: float, liquidity: float = 10_000.0) -> float:
        """Compute linear slippage cost for a given trade size and market liquidity.

        Slippage increases proportionally with trade size and inversely with
        available liquidity. This models the price impact of consuming the order book.

        Slippage cost = (size / liquidity) * size

        Args:
            size: Trade size in currency units
            liquidity: Estimated available liquidity in the market

        Returns:
            Slippage cost in currency units (non-negative)
        """
        return float((size / max(liquidity, 1.0)) * size)

    def execute(
        self,
        opportunity: Dict[str, Any],
        position_size: float,
        slippage_model: str = "linear",
    ) -> Dict[str, Any]:
        """Execute a single trade opportunity.

        Deducts position_size from cash, applies slippage, and records the trade.
        When the trade resolves (via settle), PnL is computed and cash updated.

        For backtesting purposes, the trade is immediately settled using the
        ``outcome`` field in the opportunity dict.

        Args:
            opportunity: Dict with keys:
                - market_id (str): unique identifier for the market
                - odds (float): decimal odds
                - outcome (int or float): actual binary outcome (1=win, 0=loss)
                - liquidity (float, optional): available liquidity (default 10000)
            position_size: Amount to bet in currency units
            slippage_model: Currently only 'linear' is supported

        Returns:
            Dict with keys: market_id, position_size, slippage, net_size, pnl, cash_after
        """
        market_id = opportunity.get("market_id", f"market_{len(self._trade_history)}")
        odds = float(opportunity.get("odds", 2.0))
        outcome = float(opportunity.get("outcome", 0))
        liquidity = float(opportunity.get("liquidity", 10_000.0))

        # Clip position to available cash
        position_size = min(float(position_size), self._cash)
        if position_size <= 0:
            return {"market_id": market_id, "position_size": 0.0, "slippage": 0.0, "net_size": 0.0, "pnl": 0.0, "cash_after": self._cash}

        if slippage_model == "linear":
            slippage_cost = self.compute_slippage(position_size, liquidity)
        else:
            slippage_cost = 0.0

        net_size = position_size - slippage_cost

        # Deduct full bet upfront; on a win, return net_size stake plus profit.
        self._cash -= position_size
        if outcome == 1:
            pnl = net_size * (odds - 1.0)
            self._cash += net_size + pnl  # stake returned + winnings
        else:
            pnl = -net_size  # lost the effective stake (after slippage)

        self._pnl_history.append(pnl)
        trade_record = {
            "market_id": market_id,
            "position_size": position_size,
            "slippage": slippage_cost,
            "net_size": net_size,
            "pnl": pnl,
            "cash_after": self._cash,
        }
        self._trade_history.append(trade_record)
        return trade_record

    def get_portfolio_stats(self) -> Dict[str, float]:
        """Return aggregate portfolio statistics.

        Returns:
            Dict with keys: total_pnl, n_trades, win_rate, max_drawdown, cash
        """
        if not self._pnl_history:
            return {
                "total_pnl": 0.0,
                "n_trades": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
                "cash": self._cash,
            }

        pnl_arr = np.array(self._pnl_history)
        # Only count trades where we actually bet (pnl != 0 or it's a real trade)
        n_trades = len(pnl_arr)
        wins = int((pnl_arr > 0).sum())
        win_rate = wins / n_trades if n_trades > 0 else 0.0

        equity = np.cumsum(pnl_arr) + self._initial_cash
        peak = np.maximum.accumulate(equity)
        with np.errstate(invalid="ignore", divide="ignore"):
            dd = np.where(peak > 0, (equity - peak) / peak, 0.0)
        mdd = float(np.nanmin(dd)) if len(dd) > 0 else 0.0

        return {
            "total_pnl": float(pnl_arr.sum()),
            "n_trades": n_trades,
            "win_rate": float(win_rate),
            "max_drawdown": mdd,
            "cash": float(self._cash),
        }
