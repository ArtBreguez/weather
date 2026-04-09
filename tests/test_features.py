import pytest
import numpy as np
import pandas as pd
from weather_model.data.fetchers import generate_synthetic_data
from weather_model.features.engineering import FeatureEngineer


@pytest.fixture
def sample_df():
    return generate_synthetic_data(n_days=200, seed=0)


def test_add_lags(sample_df):
    fe = FeatureEngineer()
    result = fe.add_lags(sample_df.copy(), cols=["tmax"], lags=[1, 2])
    assert "tmax_lag1" in result.columns
    assert "tmax_lag2" in result.columns
    assert result["tmax_lag1"].iloc[5] == pytest.approx(result["tmax"].iloc[4])


def test_add_rolling_stats(sample_df):
    fe = FeatureEngineer()
    result = fe.add_rolling_stats(sample_df.copy(), cols=["tmax"], windows=[7])
    assert "tmax_roll7_mean" in result.columns
    assert "tmax_roll7_std" in result.columns


def test_add_anomalies(sample_df):
    fe = FeatureEngineer()
    result = fe.add_anomalies(sample_df.copy(), col="tmax")
    assert "tmax_anomaly" in result.columns
    # Anomalies should be roughly zero-mean
    assert abs(result["tmax_anomaly"].mean()) < 5.0


def test_add_seasonality(sample_df):
    fe = FeatureEngineer()
    result = fe.add_seasonality(sample_df.copy())
    assert "doy_sin" in result.columns
    assert "doy_cos" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns
    # sin/cos should be bounded [-1, 1]
    assert result["doy_sin"].between(-1, 1).all()
    assert result["doy_cos"].between(-1, 1).all()
