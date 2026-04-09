"""Performance metrics for weather forecast evaluation and trading."""
import numpy as np
from typing import Dict, Tuple


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the Brier Score (mean squared probability error).

    Brier Score = mean((y_prob - y_true)^2)
    A perfect probabilistic forecast scores 0; random guessing scores 0.25.

    Args:
        y_true: Binary ground-truth labels (0 or 1)
        y_prob: Predicted probabilities in [0, 1]

    Returns:
        Brier score in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    """Compute the binary log-loss (cross-entropy loss).

    Log-loss = -mean(y * log(p) + (1 - y) * log(1 - p))
    Strongly penalises confident wrong predictions. Lower is better.

    Args:
        y_true: Binary ground-truth labels
        y_prob: Predicted probabilities in [0, 1]
        eps: Small value to clip probabilities away from 0 and 1

    Returns:
        Log-loss (non-negative float)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute empirical calibration curve data.

    Divides predictions into n_bins equal-width buckets and computes the
    empirical positive rate in each bucket.

    Args:
        y_true: Binary ground-truth labels
        y_prob: Predicted probabilities in [0, 1]
        n_bins: Number of histogram bins

    Returns:
        Tuple of (fraction_of_positives, mean_predicted_value) each length n_bins
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    fraction_of_positives = np.zeros(n_bins)
    mean_predicted_value = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            fraction_of_positives[i] = np.nan
            mean_predicted_value[i] = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        else:
            fraction_of_positives[i] = float(y_true[mask].mean())
            mean_predicted_value[i] = float(y_prob[mask].mean())

    return fraction_of_positives, mean_predicted_value


def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualised Sharpe ratio from a daily returns series.

    Sharpe = sqrt(252) * (mean(r) - rf) / std(r)
    Assumes daily returns and 252 trading days per year.

    Args:
        returns: Array of daily returns (e.g. PnL per trade as a fraction)
        risk_free_rate: Daily risk-free rate (default 0)

    Returns:
        Annualised Sharpe ratio
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free_rate
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.sqrt(252) * np.mean(excess) / std)


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Compute maximum peak-to-trough drawdown as a fraction.

    Returns the most negative (largest loss) peak-to-trough percentage
    decline in the equity curve. Always returns a non-positive value.

    Args:
        equity_curve: Array of cumulative equity values

    Returns:
        Maximum drawdown (non-positive float, e.g. -0.15 means -15%)
    """
    equity_curve = np.asarray(equity_curve, dtype=float)
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    # Avoid division by zero on a flat curve
    with np.errstate(invalid="ignore", divide="ignore"):
        drawdowns = np.where(peak != 0, (equity_curve - peak) / np.abs(peak), 0.0)
    return float(np.nanmin(drawdowns))


def compare_to_baseline(
    predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare model predictions to a baseline (e.g. climatological mean).

    Args:
        predictions: Model predicted probabilities
        baseline_predictions: Baseline predicted probabilities
        y_true: Binary ground-truth labels

    Returns:
        Dict with 'model' and 'baseline' keys, each containing 'brier' and 'log_loss'
    """
    return {
        "model": {
            "brier": brier_score(y_true, predictions),
            "log_loss": log_loss_score(y_true, predictions),
        },
        "baseline": {
            "brier": brier_score(y_true, baseline_predictions),
            "log_loss": log_loss_score(y_true, baseline_predictions),
        },
    }
