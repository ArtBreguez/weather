import pytest
import numpy as np
import pandas as pd
from weather_model.data.fetchers import generate_synthetic_data
from weather_model.features.engineering import FeatureEngineer
from weather_model.models.statistical import BayesianRidgeModel
from weather_model.models.ml_models import XGBoostModel, RandomForestModel
from weather_model.models.ensemble import WeightedEnsemble


@pytest.fixture
def X_y():
    df = generate_synthetic_data(n_days=300, seed=1)
    fe = FeatureEngineer()
    X, y = fe.build_features(df)
    return X, y


def test_bayesian_fit_predict(X_y):
    X, y = X_y
    model = BayesianRidgeModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    probs = model.predict_proba(X, threshold=float(np.median(y)))
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_xgboost_fit_predict(X_y):
    X, y = X_y
    model = XGBoostModel()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    probs = model.predict_proba(X, threshold=float(np.median(y)))
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_ensemble_fit_predict(X_y):
    X, y = X_y
    models = [BayesianRidgeModel(), RandomForestModel()]
    ensemble = WeightedEnsemble(models)
    ensemble.fit(X, y)
    probs = ensemble.predict_proba(X, threshold=float(np.median(y)))
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0
