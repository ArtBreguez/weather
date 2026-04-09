# Weather Forecasting for Polymarket Prediction Markets

A production-ready Python framework for building probabilistic weather forecasts
and trading them on [Polymarket](https://polymarket.com) prediction markets.

## Architecture

```
weather/
├── weather_model/
│   ├── data/
│   │   ├── fetchers.py        # NOAA CDO, GFS, ECMWF, synthetic data
│   │   └── preprocessor.py   # Cleaning, normalisation, train/test split
│   ├── features/
│   │   └── engineering.py    # Lags, rolling stats, anomalies, seasonality
│   ├── models/
│   │   ├── statistical.py    # BayesianRidge, Ridge regression
│   │   ├── ml_models.py      # XGBoost, RandomForest, MLP
│   │   ├── ensemble.py       # Weighted ensemble with Brier-score weights
│   │   └── calibration.py   # Isotonic and Platt calibration
│   ├── backtest/
│   │   ├── framework.py      # Walk-forward backtester (TimeSeriesSplit)
│   │   └── metrics.py        # Brier score, log-loss, Sharpe, max drawdown
│   ├── trading/
│   │   ├── edge.py           # Edge detection, expected value
│   │   ├── sizing.py         # Kelly criterion, risk parity, position sizer
│   │   └── execution.py      # Trade executor with slippage model
│   └── pipeline/
│       └── runner.py         # FullPipeline orchestrating all components
└── tests/
    ├── test_features.py
    ├── test_models.py
    ├── test_backtest.py
    └── test_trading.py
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Run the backtest pipeline

```python
from weather_model.pipeline.runner import FullPipeline, DEFAULT_CONFIG

pipeline = FullPipeline(DEFAULT_CONFIG)
results = pipeline.run_backtest()
print(f"Brier score : {results['brier']:.4f}")
print(f"Sharpe ratio: {results['sharpe']:.2f}")
print(f"Max drawdown: {results['max_drawdown']*100:.1f}%")
```

### Daily prediction run

```python
from weather_model.pipeline.runner import FullPipeline

config = {
    "data_source": "noaa",
    "noaa_token": "YOUR_TOKEN_HERE",
    "station_id": "GHCND:USW00094728",   # New York JFK
    "threshold": 30.0,                   # 30°C hot-day market
    "min_edge": 0.05,
}
pipeline = FullPipeline(config)
result = pipeline.run_daily()
print(result["opportunities"])
```

### Use individual components

```python
from weather_model.data.fetchers import generate_synthetic_data
from weather_model.features.engineering import FeatureEngineer
from weather_model.models.ml_models import XGBoostModel
from weather_model.trading.edge import EdgeDetector
from weather_model.trading.sizing import PositionSizer
import numpy as np

# Generate data and features
df = generate_synthetic_data(n_days=365)
fe = FeatureEngineer()
X, y = fe.build_features(df, threshold=30.0)

# Train model
model = XGBoostModel()
model.fit(X, y)
probs = model.predict_proba(X, threshold=0.5)

# Detect edge vs market
detector = EdgeDetector()
market_odds = np.full(len(probs), 2.0)
opportunities = detector.find_mispriced_markets(probs, market_odds, threshold=0.05)
print(opportunities)

# Size positions
sizer = PositionSizer(kelly_fraction=0.25)
for _, row in opportunities.iterrows():
    size = sizer.size_position(edge=row["edge"], odds=row["market_odds"], bankroll=10_000)
    print(f"  Bet ${size:.2f} on {row['direction']} at odds {row['market_odds']:.2f}")
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Temporal split | Walk-forward CV | Prevents data leakage |
| Probability output | Sigmoid / Normal CDF | Calibrated uncertainty |
| Position sizing | Fractional Kelly (25%) | Reduces variance vs full Kelly |
| Slippage model | Linear (size / liquidity) | Captures order-book impact |
| Ensemble weights | 1 / Brier score | Better-calibrated models get more weight |

## Why Polymarket Weather Markets?

- **Recency bias**: Traders over-weight recent memorable events
- **Thin liquidity**: Prices are slow to update after new GFS/ECMWF runs
- **Climatological anchor**: Most participants ignore historical base rates
- **Availability heuristic**: Memorable snowstorms inflate snow-market prices

## Running Tests

```bash
pytest tests/ -v
```

## Data Sources

| Source | Type | Notes |
|--------|------|-------|
| NOAA CDO | Historical observations | Requires free API token |
| GFS (simulated) | 10-day ensemble forecast | 21-member ensemble |
| ECMWF (simulated) | 15-day ensemble forecast | 51-member ensemble |
| Synthetic | Testing | Seasonal + noise model |
