"""Statistical weather models: Bayesian Ridge and Ridge regression wrappers."""
import abc
import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.linear_model import BayesianRidge, Ridge
from typing import Optional


class BaseWeatherModel(abc.ABC):
    """Abstract base class for all weather forecast models.

    Subclasses must implement fit, predict, and predict_proba so that the
    WeightedEnsemble and WalkForwardBacktester can treat every model uniformly.
    """

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseWeatherModel":
        """Train the model on features X and target y.

        Args:
            X: Feature DataFrame
            y: Target Series (binary 0/1 for classification tasks)

        Returns:
            self
        """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return point-estimate predictions.

        Args:
            X: Feature DataFrame

        Returns:
            1-D numpy array of predictions
        """

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Return probability that the outcome exceeds threshold.

        Args:
            X: Feature DataFrame
            threshold: Decision boundary for converting regression output to probability.
                       If None, 0.5 is used as the midpoint.

        Returns:
            1-D numpy array of probabilities in [0, 1]
        """


class BayesianRidgeModel(BaseWeatherModel):
    """Bayesian Ridge regression model with uncertainty quantification.

    BayesianRidge places a Gaussian prior over the weights and infers the
    posterior distribution analytically. The posterior variance gives a
    natural measure of prediction uncertainty, which we convert to a
    calibrated probability using the normal CDF.

    Prediction uncertainty grows for inputs far from the training distribution,
    making this model conservative (probabilities closer to 0.5) on out-of-sample
    dates - a useful property for Polymarket betting where overconfidence is costly.

    Args:
        **kwargs: Passed directly to sklearn.linear_model.BayesianRidge
    """

    def __init__(self, **kwargs):
        self._model = BayesianRidge(**kwargs)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BayesianRidgeModel":
        """Fit Bayesian Ridge by maximising the evidence lower bound.

        Args:
            X: Feature DataFrame
            y: Continuous or binary target Series

        Returns:
            self
        """
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return posterior mean predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of posterior mean predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Convert Bayesian posterior to probability using normal CDF.

        P(y > threshold | X) = 1 - Phi((threshold - mu) / sigma)
        where mu is the posterior mean and sigma is the posterior standard deviation.
        This correctly propagates model uncertainty into the probability output.

        Args:
            X: Feature DataFrame
            threshold: Value to compute exceedance probability for.
                       Defaults to 0.5 (assumes binary target in [0, 1]).

        Returns:
            Array of exceedance probabilities in [0, 1]
        """
        if threshold is None:
            threshold = 0.5
        mu, sigma = self._model.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-8)
        probs = 1.0 - ndtr((threshold - mu) / sigma)
        return np.clip(probs, 0.0, 1.0)


class LinearRegressionModel(BaseWeatherModel):
    """Ridge regression model (L2-regularised ordinary least squares).

    Ridge regression shrinks coefficients towards zero, reducing overfitting
    on high-dimensional weather feature sets. The regularisation strength alpha
    is a hyperparameter that should be tuned via cross-validation.

    Probabilities are obtained by applying the logistic sigmoid to the raw
    regression score, scaled relative to the threshold.

    Args:
        alpha: L2 regularisation strength (default 1.0)
        **kwargs: Additional keyword arguments passed to sklearn Ridge
    """

    def __init__(self, alpha: float = 1.0, **kwargs):
        self._model = Ridge(alpha=alpha, **kwargs)
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        """Fit Ridge regression via closed-form normal equations.

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
        """Return Ridge regression predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Convert regression output to probability via logistic sigmoid.

        The sigmoid is applied to the difference (prediction - threshold) so
        that predictions at the threshold map to probability 0.5.

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
