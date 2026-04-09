"""Machine learning weather models: XGBoost, Random Forest, and MLP."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from typing import Optional

from .statistical import BaseWeatherModel


class XGBoostModel(BaseWeatherModel):
    """Gradient-boosted tree model using XGBoost.

    XGBoost sequentially adds decision trees that correct the residuals of
    the ensemble so far. It handles non-linear interactions between weather
    variables and is robust to the heavy-tailed distributions common in
    precipitation and extreme temperature data.

    Feature importances are available after fitting, which helps identify
    which lags and rolling statistics carry the most predictive signal.

    Args:
        n_estimators: Number of boosting rounds
        max_depth: Maximum depth of each tree
        learning_rate: Step size shrinkage to prevent overfitting
        **kwargs: Additional keyword arguments passed to XGBRegressor
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        **kwargs,
    ):
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            verbosity=0,
            **kwargs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        """Train XGBoost regressor on weather features.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            self
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return XGBoost point predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Convert regression output to probability via logistic sigmoid.

        The sigmoid maps (prediction - threshold) to [0, 1]. Values well above
        the threshold map to probabilities near 1; values well below map near 0.

        Args:
            X: Feature DataFrame
            threshold: Decision boundary. Defaults to 0.5.

        Returns:
            Array of probabilities in [0, 1]
        """
        if threshold is None:
            threshold = 0.5
        raw = self._model.predict(X)
        probs = 1.0 / (1.0 + np.exp(-(raw - threshold)))
        return np.clip(probs, 0.0, 1.0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from XGBoost (gain-based).

        Returns:
            Array of feature importance scores
        """
        return self._model.feature_importances_


class RandomForestModel(BaseWeatherModel):
    """Random Forest ensemble of decision trees.

    Random forests average many decorrelated trees, reducing variance without
    increasing bias. The bootstrap sampling and random feature selection make
    the model robust to noisy weather measurements.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of each tree (None means unlimited)
        **kwargs: Additional keyword arguments passed to RandomForestRegressor
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 8,
        **kwargs,
    ):
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            **kwargs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Train Random Forest on weather features.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            self
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return Random Forest point predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Convert regression output to probability via logistic sigmoid.

        Args:
            X: Feature DataFrame
            threshold: Decision boundary. Defaults to 0.5.

        Returns:
            Array of probabilities in [0, 1]
        """
        if threshold is None:
            threshold = 0.5
        raw = self._model.predict(X)
        probs = 1.0 / (1.0 + np.exp(-(raw - threshold)))
        return np.clip(probs, 0.0, 1.0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from mean decrease in impurity.

        Returns:
            Array of feature importance scores
        """
        return self._model.feature_importances_


class MLPModel(BaseWeatherModel):
    """Multi-layer perceptron regressor using scikit-learn.

    The MLP can learn non-linear feature interactions that tree-based models
    might miss. It works best when features are normalised, so the DataPreprocessor
    should be applied before passing data to this model.

    Args:
        hidden_layer_sizes: Tuple of hidden layer sizes
        max_iter: Maximum training iterations
        **kwargs: Additional keyword arguments passed to MLPRegressor
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32),
        max_iter: int = 500,
        **kwargs,
    ):
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42,
            **kwargs,
        )
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MLPModel":
        """Train MLP regressor using backpropagation.

        Args:
            X: Feature DataFrame (should be normalised)
            y: Target Series

        Returns:
            self
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return MLP point predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Convert regression output to probability via logistic sigmoid.

        Args:
            X: Feature DataFrame
            threshold: Decision boundary. Defaults to 0.5.

        Returns:
            Array of probabilities in [0, 1]
        """
        if threshold is None:
            threshold = 0.5
        raw = self._model.predict(X)
        probs = 1.0 / (1.0 + np.exp(-(raw - threshold)))
        return np.clip(probs, 0.0, 1.0)
