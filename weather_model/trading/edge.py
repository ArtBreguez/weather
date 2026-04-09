"""Edge detection utilities for Polymarket weather prediction markets."""
import numpy as np
import pandas as pd
from typing import Union


class EdgeDetector:
    """Detects mispricings between model probabilities and market odds.

    Polymarket weather markets are frequently mispriced because:
    - Retail participants exhibit recency bias (over-weighting recent extreme events).
    - Availability heuristic causes traders to anchor on memorable weather events.
    - Thin liquidity means market prices are slow to update when new forecast data
      becomes available; a 6 AM GFS update may not be reflected in prices until hours later.
    - Most participants do not adjust for base-rate climatology, so markets for
      rare events (e.g. snow in April) are systematically over-priced.

    The edge is simply the difference between the model's probability estimate and
    the market-implied probability (1 / decimal_odds).
    """

    def compute_edge(
        self,
        model_prob: Union[float, np.ndarray],
        market_prob: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute the edge: model probability minus market-implied probability.

        A positive edge means the model thinks the event is more likely than the
        market implies, suggesting a BUY bet. A negative edge suggests a SELL/NO bet.

        Args:
            model_prob: Model's estimated probability of the event
            market_prob: Market-implied probability (1 / decimal_odds)

        Returns:
            Edge in [-1, 1]
        """
        return np.asarray(model_prob) - np.asarray(market_prob)

    def find_mispriced_markets(
        self,
        predictions: np.ndarray,
        market_odds: np.ndarray,
        threshold: float = 0.05,
    ) -> pd.DataFrame:
        """Identify markets where the edge exceeds the threshold.

        Only markets with |edge| > threshold are returned, as smaller mispricings
        may be within the noise of model uncertainty and transaction costs.

        Args:
            predictions: Model probability estimates (array of floats in [0, 1])
            market_odds: Decimal market odds for each market (e.g. 2.0 = evens)
            threshold: Minimum absolute edge to flag a market as mispriced

        Returns:
            DataFrame with columns: model_prob, market_odds, implied_prob, edge, direction
        """
        predictions = np.asarray(predictions, dtype=float)
        market_odds = np.asarray(market_odds, dtype=float)
        implied_prob = 1.0 / np.maximum(market_odds, 1.001)
        edge = self.compute_edge(predictions, implied_prob)
        mask = np.abs(edge) > threshold
        df = pd.DataFrame({
            "model_prob": predictions[mask],
            "market_odds": market_odds[mask],
            "implied_prob": implied_prob[mask],
            "edge": edge[mask],
            "direction": np.where(edge[mask] > 0, "BUY", "SELL"),
        })
        return df.reset_index(drop=True)

    def expected_value(
        self,
        model_prob: Union[float, np.ndarray],
        market_odds: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Compute expected value of a unit bet at decimal odds.

        EV = model_prob * (odds - 1) - (1 - model_prob) * 1
           = model_prob * odds - 1

        A positive EV indicates a profitable bet in expectation.

        Args:
            model_prob: Model's estimated probability of winning
            market_odds: Decimal market odds (e.g. 2.0 pays back double stake)

        Returns:
            Expected value of a unit bet
        """
        model_prob = np.asarray(model_prob, dtype=float)
        market_odds = np.asarray(market_odds, dtype=float)
        return model_prob * (market_odds - 1.0) - (1.0 - model_prob)
