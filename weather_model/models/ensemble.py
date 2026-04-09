"""Weighted ensemble of BaseWeatherModel instances."""
import numpy as np
import pandas as pd
from typing import List, Optional

from .statistical import BaseWeatherModel
from ..backtest.metrics import brier_score


class WeightedEnsemble:
    """Ensemble that combines multiple BaseWeatherModel instances via learned weights.

    Weights are learned inversely proportional to each model's Brier score on a
    held-out validation portion of the training data. Models with better-calibrated
    probabilities receive higher weight. The final probability is a weighted average
    of each model's predict_proba output.

    Using an ensemble reduces variance and typically improves calibration compared
    to any single model. For Polymarket markets, better calibration means fewer
    systematic over/under-pricing errors.

    Args:
        models: List of BaseWeatherModel instances
        val_frac: Fraction of training data to use for weight learning
    """

    def __init__(self, models: List[BaseWeatherModel], val_frac: float = 0.2):
        self._models = models
        self._val_frac = val_frac
        self._weights: np.ndarray = np.ones(len(models)) / len(models)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WeightedEnsemble":
        """Fit each base model and learn ensemble weights from validation Brier scores.

        The training set is split temporally: the first (1 - val_frac) portion
        trains the base models and the remaining portion computes Brier scores
        used to set ensemble weights.

        Args:
            X: Feature DataFrame
            y: Binary target Series

        Returns:
            self
        """
        n = len(X)
        cutoff = int(n * (1 - self._val_frac))
        X_train = X.iloc[:cutoff]
        y_train = y.iloc[:cutoff]
        X_val = X.iloc[cutoff:]
        y_val = y.iloc[cutoff:]

        threshold = float(y_train.median()) if y_train.nunique() > 2 else 0.5

        brier_scores = []
        for model in self._models:
            model.fit(X_train, y_train)
            if len(X_val) > 0:
                probs = model.predict_proba(X_val, threshold=threshold)
                bs = brier_score(y_val.values, probs)
            else:
                bs = 1.0
            # Avoid division by zero
            brier_scores.append(max(bs, 1e-9))

        inv_brier = 1.0 / np.array(brier_scores)
        self._weights = inv_brier / inv_brier.sum()
        return self

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Return weighted average of base model probabilities.

        Args:
            X: Feature DataFrame
            threshold: Passed through to each base model's predict_proba.
                       Can be None; each model uses its own default.

        Returns:
            Array of ensemble probabilities in [0, 1]
        """
        probs = np.zeros(len(X))
        for model, weight in zip(self._models, self._weights):
            probs += weight * model.predict_proba(X, threshold=threshold)
        return np.clip(probs, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return weighted average of base model point predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of ensemble predictions
        """
        preds = np.zeros(len(X))
        for model, weight in zip(self._models, self._weights):
            preds += weight * model.predict(X)
        return preds

    @property
    def weights(self) -> np.ndarray:
        """Current ensemble weights (one per model)."""
        return self._weights.copy()
