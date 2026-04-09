import pytest
import numpy as np
import pandas as pd
from weather_model.data.fetchers import generate_synthetic_data
from weather_model.features.engineering import FeatureEngineer
from weather_model.models.statistical import BayesianRidgeModel
from weather_model.backtest.framework import WalkForwardBacktester
from weather_model.backtest.metrics import brier_score, log_loss_score, sharpe_ratio, max_drawdown


@pytest.fixture
def backtest_data():
    df = generate_synthetic_data(n_days=400, seed=2)
    fe = FeatureEngineer()
    X, y = fe.build_features(df)
    return X, y


def test_walk_forward(backtest_data):
    X, y = backtest_data
    model = BayesianRidgeModel()
    market_odds = np.full(len(y), 2.0)
    bt = WalkForwardBacktester(model, n_splits=3, gap=3)
    results = bt.run(X, y, market_odds)
    assert "predictions" in results
    assert "actuals" in results
    assert "pnl" in results
    assert len(results["predictions"]) > 0


def test_brier_score():
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    bs = brier_score(y_true, y_prob)
    assert 0.0 <= bs <= 1.0


def test_sharpe_ratio():
    returns = np.random.default_rng(0).normal(0.01, 0.05, 252)
    sr = sharpe_ratio(returns)
    assert isinstance(sr, float)


def test_max_drawdown():
    equity = np.array([100, 110, 105, 95, 100, 120])
    dd = max_drawdown(equity)
    assert dd < 0  # drawdown is negative
