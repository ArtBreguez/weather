"""Position sizing: Kelly Criterion, Risk Parity, and composite PositionSizer."""
import numpy as np
from scipy.optimize import minimize
from typing import Optional


class KellyCriterion:
    """Kelly Criterion position sizing for binary prediction markets.

    The Kelly Criterion maximizes the long-run growth rate of the bankroll by
    betting a fraction proportional to the edge. Full Kelly is theoretically
    optimal but highly volatile; fractional Kelly (25%) is used in practice to
    reduce variance and guard against model error.

    The Kelly fraction for a binary bet is: f = edge / (odds - 1)
    where edge = model_prob - implied_prob.
    """

    def full_kelly(self, edge: float, odds: float) -> float:
        """Compute the full Kelly fraction of bankroll to bet.

        f* = edge / (odds - 1)

        Args:
            edge: Model edge = model_prob - 1/odds (can be negative)
            odds: Decimal odds (must be > 1)

        Returns:
            Kelly fraction clamped to [0, 1]
        """
        if odds <= 1.0:
            return 0.0
        fraction = edge / (odds - 1.0)
        return float(np.clip(fraction, 0.0, 1.0))

    def fractional_kelly(self, edge: float, odds: float, fraction: float = 0.25) -> float:
        """Compute fractional Kelly to reduce variance.

        Multiplies full Kelly by ``fraction``. Using 25% Kelly reduces the
        standard deviation of returns by 75% while capturing most of the
        expected growth.

        Args:
            edge: Model edge
            odds: Decimal odds
            fraction: Multiplier applied to full Kelly (default 0.25)

        Returns:
            Fractional Kelly bet size in [0, fraction]
        """
        return fraction * self.full_kelly(edge, odds)

    def kelly_with_uncertainty(
        self,
        edge: float,
        odds: float,
        model_uncertainty: float,
    ) -> float:
        """Reduce Kelly fraction by a model uncertainty factor.

        When the model reports high uncertainty (e.g. Bayesian posterior variance
        is large), we should bet less. The uncertainty factor shrinks the Kelly
        fraction: the higher the uncertainty, the closer the bet is to zero.

        Args:
            edge: Model edge
            odds: Decimal odds
            model_uncertainty: Value in [0, 1] where 0 means certain, 1 means maximal uncertainty

        Returns:
            Uncertainty-adjusted Kelly fraction in [0, 1]
        """
        model_uncertainty = float(np.clip(model_uncertainty, 0.0, 1.0))
        uncertainty_factor = 1.0 - model_uncertainty
        return self.full_kelly(edge, odds) * uncertainty_factor


class RiskParity:
    """Risk parity portfolio weighting using equal risk contribution.

    Risk parity assigns weights such that each position contributes equally to
    total portfolio variance. This is useful when running multiple simultaneous
    weather market positions with different volatilities.
    """

    def compute_weights(self, returns_cov_matrix: np.ndarray) -> np.ndarray:
        """Compute risk parity weights by minimising deviation from equal risk contribution.

        Uses scipy.optimize to find the weight vector w such that each asset's
        marginal risk contribution (w_i * (Sigma @ w)_i) is equal.

        Args:
            returns_cov_matrix: Square covariance matrix of asset returns (n x n)

        Returns:
            Weight vector of length n summing to 1
        """
        n = returns_cov_matrix.shape[0]
        cov = np.asarray(returns_cov_matrix, dtype=float)

        def risk_contributions_variance(weights: np.ndarray) -> float:
            """Objective: minimise variance of risk contributions."""
            w = weights / weights.sum()
            sigma = cov @ w
            rc = w * sigma
            total_rc = rc.sum()
            if total_rc == 0:
                return 0.0
            rc_norm = rc / total_rc
            target = 1.0 / n
            return float(np.sum((rc_norm - target) ** 2))

        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        result = minimize(
            risk_contributions_variance,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        weights = np.clip(result.x, 0.0, 1.0)
        total = weights.sum()
        if total == 0:
            return np.ones(n) / n
        return weights / total


class PositionSizer:
    """Composite position sizer combining Kelly Criterion with bankroll constraint.

    Ensures that no single bet can violate the maximum drawdown limit by
    capping the position size at max_drawdown_limit * bankroll.

    Args:
        kelly_fraction: Fractional Kelly multiplier (default 0.25)
        max_drawdown_limit: Maximum fraction of bankroll to risk per trade (default 0.2)
    """

    def __init__(self, kelly_fraction: float = 0.25, max_drawdown_limit: float = 0.20):
        self._kelly = KellyCriterion()
        self._kelly_fraction = kelly_fraction
        self._max_drawdown_limit = max_drawdown_limit

    def size_position(
        self,
        edge: float,
        odds: float,
        bankroll: float,
        max_drawdown_limit: Optional[float] = None,
    ) -> float:
        """Compute position size in currency units.

        Applies fractional Kelly to determine the optimal fraction, then caps
        the resulting dollar amount at max_drawdown_limit * bankroll.

        Args:
            edge: Model edge (model_prob - implied_prob)
            odds: Decimal market odds
            bankroll: Current bankroll in currency units
            max_drawdown_limit: Override for maximum fraction at risk (default from __init__)

        Returns:
            Position size in currency units (non-negative)
        """
        limit = max_drawdown_limit if max_drawdown_limit is not None else self._max_drawdown_limit
        kelly_frac = self._kelly.fractional_kelly(edge, odds, fraction=self._kelly_fraction)
        raw_size = kelly_frac * bankroll
        max_size = limit * bankroll
        return float(min(raw_size, max_size))
