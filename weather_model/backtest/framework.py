"""Walk-forward backtesting framework for weather prediction models."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Any

from ..models.statistical import BaseWeatherModel


class WalkForwardBacktester:
    """Walk-forward backtester using sklearn TimeSeriesSplit.

    Walk-forward validation trains on all data up to a point in time and
    tests on the immediately following period, then rolls the window forward.
    This faithfully simulates live deployment where no future data is ever
    available during training.

    A gap of ``gap`` time steps is left between the training end and test start
    to prevent leakage through lagged features that look back many days.

    Each fold simulates trading: if the model's edge exceeds zero, a unit bet
    is placed at the market odds. 0.5 % slippage is applied to each trade to
    reflect the cost of crossing the bid-ask spread on Polymarket.

    Args:
        model: A BaseWeatherModel instance (will be re-fit on each fold)
        n_splits: Number of walk-forward folds
        gap: Number of time steps to skip between train and test to prevent leakage
    """

    SLIPPAGE = 0.005  # 0.5 % per trade

    def __init__(self, model: BaseWeatherModel, n_splits: int = 5, gap: int = 7):
        self._model = model
        self._n_splits = n_splits
        self._gap = gap

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        market_odds: np.ndarray,
    ) -> Dict[str, Any]:
        """Execute the walk-forward backtest.

        For each fold:
        1. Train model on training slice.
        2. Predict probabilities on test slice.
        3. Compute edge = model_prob - (1 / market_odds).
        4. If edge > 0, place a unit bet (1.0) at market odds with slippage.
        5. Compute per-trade PnL.

        Args:
            X: Feature DataFrame (index must be contiguous integers)
            y: Binary target Series aligned with X
            market_odds: Array of decimal market odds aligned with X

        Returns:
            Dict with keys:
                predictions: all out-of-sample probabilities
                actuals: corresponding binary outcomes
                market_odds: corresponding market odds
                pnl: per-trade PnL list
                drawdown_series: equity-curve drawdown array
                edge_per_trade: edge at time of each bet
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        market_odds = np.asarray(market_odds)

        tscv = TimeSeriesSplit(n_splits=self._n_splits, gap=self._gap)

        all_preds: list = []
        all_actuals: list = []
        all_odds: list = []
        all_pnl: list = []
        all_edges: list = []

        threshold = float(y.median()) if y.nunique() > 2 else 0.5

        for train_idx, test_idx in tscv.split(X):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            odds_test = market_odds[test_idx]

            self._model.fit(X_train, y_train)
            probs = self._model.predict_proba(X_test, threshold=threshold)
            implied_prob = 1.0 / np.maximum(odds_test, 1.001)
            edges = probs - implied_prob

            for prob, actual, odd, edge in zip(probs, y_test.values, odds_test, edges):
                all_preds.append(float(prob))
                all_actuals.append(float(actual))
                all_odds.append(float(odd))
                all_edges.append(float(edge))

                if edge > 0:
                    effective_odds = odd * (1.0 - self.SLIPPAGE)
                    pnl = (effective_odds - 1.0) * 1.0 if actual == 1 else -1.0
                else:
                    pnl = 0.0
                all_pnl.append(pnl)

        equity = np.cumsum(all_pnl)
        drawdown_series = self._compute_drawdown(equity)

        return {
            "predictions": np.array(all_preds),
            "actuals": np.array(all_actuals),
            "market_odds": np.array(all_odds),
            "pnl": np.array(all_pnl),
            "drawdown_series": drawdown_series,
            "edge_per_trade": np.array(all_edges),
        }

    @staticmethod
    def _compute_drawdown(equity: np.ndarray) -> np.ndarray:
        """Compute the running drawdown from peak equity.

        Args:
            equity: Cumulative PnL curve

        Returns:
            Array of drawdown values (non-positive)
        """
        if len(equity) == 0:
            return np.array([])
        peak = np.maximum.accumulate(equity)
        drawdown = equity - peak
        return drawdown
