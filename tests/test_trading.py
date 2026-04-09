import pytest
import numpy as np
import pandas as pd
from weather_model.trading.edge import EdgeDetector
from weather_model.trading.sizing import KellyCriterion, PositionSizer
from weather_model.trading.execution import TradeExecutor


def test_edge_detection():
    detector = EdgeDetector()
    edge = detector.compute_edge(0.7, 0.5)
    assert abs(edge - 0.2) < 1e-9
    ev = detector.expected_value(0.7, 2.0)
    # 0.7 * (2.0 - 1) - 0.3 = 0.4
    assert abs(ev - 0.4) < 1e-9


def test_kelly_criterion():
    kelly = KellyCriterion()
    fk = kelly.full_kelly(edge=0.1, odds=2.0)
    assert 0.0 <= fk <= 1.0
    frac = kelly.fractional_kelly(edge=0.1, odds=2.0, fraction=0.25)
    assert abs(frac - fk * 0.25) < 1e-9


def test_position_sizing():
    sizer = PositionSizer(kelly_fraction=0.25)
    size = sizer.size_position(edge=0.1, odds=2.0, bankroll=10000.0)
    assert size >= 0.0
    assert size <= 10000.0 * 0.20  # max drawdown limit


def test_mispriced_markets():
    detector = EdgeDetector()
    predictions = np.array([0.7, 0.5, 0.3, 0.6])
    market_odds = np.array([2.5, 2.0, 2.0, 1.8])
    df = detector.find_mispriced_markets(predictions, market_odds, threshold=0.05)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1
