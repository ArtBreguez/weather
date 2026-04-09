"""End-to-end pipeline runner for daily and backtest modes."""
import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from ..data.fetchers import NOAAFetcher, GFSFetcher, ECMWFFetcher, generate_synthetic_data
from ..data.preprocessor import DataPreprocessor
from ..features.engineering import FeatureEngineer
from ..models.statistical import BayesianRidgeModel, LinearRegressionModel
from ..models.ml_models import XGBoostModel, RandomForestModel, MLPModel
from ..models.ensemble import WeightedEnsemble
from ..models.calibration import calibrate_probabilities
from ..backtest.framework import WalkForwardBacktester
from ..backtest.metrics import brier_score, sharpe_ratio, max_drawdown
from ..trading.edge import EdgeDetector
from ..trading.sizing import PositionSizer
from ..trading.execution import TradeExecutor

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "data_source": "synthetic",
    "station_id": "GHCND:USW00094728",
    "noaa_token": "",
    "lat": 40.7128,
    "lon": -74.0060,
    "n_history_days": 365,
    "target_col": "tmax",
    "threshold": 30.0,
    "model_types": ["bayesian", "xgboost", "random_forest"],
    "ensemble_weights": "auto",
    "kelly_fraction": 0.25,
    "max_drawdown_limit": 0.20,
    "slippage_bps": 50,
    "min_edge": 0.05,
}

_MODEL_MAP = {
    "bayesian": BayesianRidgeModel,
    "linear": LinearRegressionModel,
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "mlp": MLPModel,
}


class FullPipeline:
    """Orchestrates the complete weather forecasting and trading pipeline.

    The pipeline proceeds in the following stages:
    1. Data ingestion  - fetch raw weather observations or forecasts
    2. Preprocessing   - clean outliers, normalise, split
    3. Feature engineering - lags, rolling stats, seasonality, anomalies
    4. Model training  - fit each model type specified in config
    5. Probability calibration - isotonic regression on held-out data
    6. Edge detection  - identify mispriced Polymarket markets
    7. Position sizing - Kelly criterion with bankroll constraints
    8. Trade execution - simulated execution with slippage

    Args:
        config: Configuration dict. Missing keys fall back to DEFAULT_CONFIG.
    """

    def __init__(self, config: Dict[str, Any] = None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}
        self._cfg = cfg

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        self._preprocessor = DataPreprocessor()
        self._feature_engineer = FeatureEngineer()
        self._edge_detector = EdgeDetector()
        self._position_sizer = PositionSizer(
            kelly_fraction=cfg["kelly_fraction"],
            max_drawdown_limit=cfg["max_drawdown_limit"],
        )
        self._executor = TradeExecutor(initial_cash=10_000.0)

        self._models = self._build_models(cfg["model_types"])
        self._ensemble = WeightedEnsemble(list(self._models.values()))

    def _build_models(self, model_types: list) -> Dict[str, Any]:
        models = {}
        for name in model_types:
            cls = _MODEL_MAP.get(name)
            if cls is None:
                logger.warning("Unknown model type '%s'; skipping.", name)
                continue
            models[name] = cls()
        return models

    def _fetch_data(self, n_days: int) -> pd.DataFrame:
        source = self._cfg["data_source"]
        if source == "noaa":
            fetcher = NOAAFetcher(api_token=self._cfg.get("noaa_token", ""))
            from datetime import datetime, timedelta
            end = datetime.today().strftime("%Y-%m-%d")
            start = (datetime.today() - timedelta(days=n_days)).strftime("%Y-%m-%d")
            return fetcher.fetch(self._cfg["station_id"], start, end)
        elif source == "gfs":
            fetcher = GFSFetcher()
            return fetcher.fetch(self._cfg["lat"], self._cfg["lon"], forecast_days=n_days)
        elif source == "ecmwf":
            fetcher = ECMWFFetcher()
            return fetcher.fetch(self._cfg["lat"], self._cfg["lon"], forecast_days=min(n_days, 15))
        else:
            return generate_synthetic_data(n_days=n_days)

    def run_daily(self) -> Dict[str, Any]:
        """Run the daily prediction pipeline.

        Fetches the latest data, builds features, generates ensemble probabilities,
        detects edges, sizes positions, and simulates execution.

        Returns:
            Dict with keys: probabilities, edges, opportunities, portfolio_stats
        """
        logger.info("Starting daily pipeline run.")
        try:
            df = self._fetch_data(self._cfg["n_history_days"])
            df = self._preprocessor.clean(df)
            threshold = self._cfg.get("threshold", float(df[self._cfg["target_col"]].median()))
            X, y = self._feature_engineer.build_features(df, threshold=threshold)

            train_df, _ = self._preprocessor.split_train_test(
                pd.concat([X, y.rename("target")], axis=1)
            )
            X_train = train_df.drop(columns=["target"])
            y_train = train_df["target"]

            self._ensemble.fit(X_train, y_train)
            probs = self._ensemble.predict_proba(X, threshold=threshold)

            # Simulate market odds as 1 / clim_prob with slight spread
            rng = np.random.default_rng(0)
            market_odds = 1.0 / np.clip(
                rng.normal(0.5, 0.1, len(probs)), 0.1, 0.95
            )

            opportunities = self._edge_detector.find_mispriced_markets(
                probs, market_odds, threshold=self._cfg["min_edge"]
            )

            self._executor.reset()
            for _, row in opportunities.iterrows():
                edge = row["edge"]
                odds = row["market_odds"]
                size = self._position_sizer.size_position(
                    edge=edge,
                    odds=odds,
                    bankroll=self._executor._cash,
                )
                # Simulate with random outcome for daily mode
                # Sample a simulated outcome from the model probability (Bernoulli trial)
                rng2 = np.random.default_rng(int(_ * 1000))
                sim_outcome = int(rng2.random() < row["model_prob"])
                opp = {"market_id": f"daily_{_}", "odds": odds, "outcome": sim_outcome}
                self._executor.execute(opp, size)

            stats = self._executor.get_portfolio_stats()
            logger.info("Daily run complete. Trades: %d", stats["n_trades"])
            return {
                "probabilities": probs,
                "edges": self._edge_detector.compute_edge(
                    probs, 1.0 / np.maximum(market_odds, 1.001)
                ),
                "opportunities": opportunities,
                "portfolio_stats": stats,
            }
        except Exception as exc:
            logger.error("Daily pipeline failed: %s", exc, exc_info=True)
            raise

    def run_backtest(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Run walk-forward backtest on synthetic or historical data.

        Generates a synthetic history of n_history_days, builds features,
        then runs WalkForwardBacktester with the ensemble model. Returns
        full backtest statistics including Sharpe ratio and max drawdown.

        Args:
            start_date: Ignored for synthetic data; kept for API consistency.
            end_date: Ignored for synthetic data; kept for API consistency.

        Returns:
            Dict with keys: backtest_results, brier, sharpe, max_drawdown
        """
        logger.info("Starting backtest pipeline run.")
        try:
            df = generate_synthetic_data(n_days=self._cfg["n_history_days"])
            df = self._preprocessor.clean(df)
            threshold = self._cfg.get("threshold", float(df[self._cfg["target_col"]].median()))
            X, y = self._feature_engineer.build_features(df, threshold=threshold)

            market_odds = np.full(len(y), 2.0)
            bt = WalkForwardBacktester(self._ensemble, n_splits=5, gap=7)
            results = bt.run(X, y, market_odds)

            actuals = results["actuals"]
            preds = results["predictions"]
            bs = brier_score(actuals, preds) if len(actuals) > 0 else float("nan")
            sr = sharpe_ratio(results["pnl"]) if len(results["pnl"]) > 0 else float("nan")
            equity = np.cumsum(results["pnl"]) + 10_000.0
            mdd = max_drawdown(equity) if len(equity) > 0 else 0.0

            logger.info(
                "Backtest complete. Brier=%.4f  Sharpe=%.2f  MaxDD=%.2f%%",
                bs,
                sr,
                mdd * 100,
            )
            return {
                "backtest_results": results,
                "brier": bs,
                "sharpe": sr,
                "max_drawdown": mdd,
            }
        except Exception as exc:
            logger.error("Backtest pipeline failed: %s", exc, exc_info=True)
            raise
