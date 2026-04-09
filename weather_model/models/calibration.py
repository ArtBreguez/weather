"""Probability calibration utilities for weather forecast models."""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from typing import Tuple


class IsotonicCalibrator:
    """Calibrate raw model probabilities using isotonic regression.

    Isotonic regression fits a non-decreasing step function that maps raw
    (uncalibrated) scores to empirical frequencies. It is non-parametric and
    can correct arbitrary monotone miscalibration, making it more flexible than
    Platt scaling but requiring more calibration samples.

    Args:
        out_of_bounds: How to handle predictions outside the training range
                       ('clip' is safe for probabilities)
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self._calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit isotonic regression on raw probabilities and binary outcomes.

        Args:
            raw_probs: Uncalibrated model scores in [0, 1]
            y_true: Binary ground-truth labels

        Returns:
            self
        """
        self._calibrator.fit(raw_probs, y_true)
        self._fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to new raw probabilities.

        Args:
            raw_probs: Uncalibrated scores in [0, 1]

        Returns:
            Calibrated probabilities in [0, 1]
        """
        return np.clip(self._calibrator.predict(raw_probs), 0.0, 1.0)


class PlattCalibrator:
    """Calibrate raw model probabilities using Platt scaling (logistic regression).

    Platt scaling fits a single-feature logistic regression that maps raw scores
    to calibrated probabilities. It is parametric and requires fewer calibration
    samples than isotonic regression, but assumes a sigmoid relationship between
    the raw score and the true probability.

    Args:
        C: Inverse of regularisation strength for LogisticRegression
    """

    def __init__(self, C: float = 1.0):
        self._calibrator = LogisticRegression(C=C)
        self._fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        """Fit logistic regression on raw probabilities.

        Args:
            raw_probs: Uncalibrated model scores
            y_true: Binary ground-truth labels

        Returns:
            self
        """
        self._calibrator.fit(raw_probs.reshape(-1, 1), y_true)
        self._fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply fitted Platt scaling to new raw probabilities.

        Args:
            raw_probs: Uncalibrated scores

        Returns:
            Calibrated probabilities in [0, 1]
        """
        return np.clip(
            self._calibrator.predict_proba(raw_probs.reshape(-1, 1))[:, 1],
            0.0,
            1.0,
        )


def calibrate_probabilities(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> np.ndarray:
    """Convenience function to fit and apply a calibrator in one step.

    Uses leave-one-out style: fits on the full data then returns in-sample
    calibrated probabilities. For production use, fit on a held-out calibration
    set to avoid overfitting.

    Args:
        raw_probs: Uncalibrated model scores
        y_true: Binary ground-truth labels
        method: 'isotonic' or 'platt'

    Returns:
        Calibrated probabilities in [0, 1]
    """
    if method == "isotonic":
        cal = IsotonicCalibrator()
    elif method == "platt":
        cal = PlattCalibrator()
    else:
        raise ValueError(f"Unknown calibration method '{method}'. Use 'isotonic' or 'platt'.")
    cal.fit(raw_probs, y_true)
    return cal.calibrate(raw_probs)


def reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute reliability diagram data (calibration curve).

    Bins predictions into n_bins equal-width probability buckets and computes
    the empirical positive rate in each bin. A perfectly calibrated model lies
    on the diagonal (fraction_of_positives == mean_predicted_value).

    Args:
        y_true: Binary ground-truth labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for the calibration curve

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) each of length n_bins
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high)
        if mask.sum() == 0:
            fraction_of_positives[i] = np.nan
            mean_predicted_value[i] = (low + high) / 2.0
        else:
            fraction_of_positives[i] = y_true[mask].mean()
            mean_predicted_value[i] = y_prob[mask].mean()

    return fraction_of_positives, mean_predicted_value
