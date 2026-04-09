"""Weather feature engineering: lags, rolling stats, anomalies, seasonality."""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class FeatureEngineer:
    """Builds predictive features from raw weather time series data.

    All transform methods operate on copies of the input DataFrame to avoid
    side effects. build_features is the main entry point that applies every
    transform and returns a clean (X, y) pair ready for model training.

    Lagged features capture autocorrelation: temperature today predicts tomorrow.
    Rolling statistics summarise recent trends and variability.
    Climatological anomalies remove seasonal mean and isolate the weather signal.
    Fourier seasonality features give models a smooth representation of the annual cycle.
    """

    def add_lags(
        self,
        df: pd.DataFrame,
        cols: List[str],
        lags: List[int] = None,
    ) -> pd.DataFrame:
        """Add lagged features for each column × lag combination.

        For example, tmax_lag1 is yesterday's maximum temperature, which is
        highly informative for predicting today's or tomorrow's temperature.

        Args:
            df: Input weather DataFrame
            cols: Column names to lag
            lags: List of integer lag values (days)

        Returns:
            DataFrame with new lag columns appended
        """
        if lags is None:
            lags = [1, 2, 3, 7, 14]
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            for lag in lags:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
        return df

    def add_rolling_stats(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int] = None,
    ) -> pd.DataFrame:
        """Add rolling mean, std, min, and max for each column × window.

        Rolling statistics capture medium-term trends. A 30-day rolling mean
        smooths day-to-day noise and reveals persistent warm or cold spells.

        Args:
            df: Input weather DataFrame
            cols: Column names to compute rolling stats for
            windows: List of window sizes in days

        Returns:
            DataFrame with new rolling stat columns appended
        """
        if windows is None:
            windows = [7, 14, 30]
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            for w in windows:
                df[f"{col}_roll{w}_mean"] = df[col].rolling(w, min_periods=1).mean()
                df[f"{col}_roll{w}_std"] = df[col].rolling(w, min_periods=2).std().fillna(0.0)
                df[f"{col}_roll{w}_min"] = df[col].rolling(w, min_periods=1).min()
                df[f"{col}_roll{w}_max"] = df[col].rolling(w, min_periods=1).max()
        return df

    def add_anomalies(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add climatological anomaly: value minus same day-of-year historical mean.

        Anomalies isolate departures from the seasonal norm. A positive tmax anomaly
        means it is warmer than average for that time of year, which is more
        informative than raw temperature for predicting extreme events.

        Args:
            df: Input weather DataFrame with a 'date' column
            col: Column name to compute anomaly for

        Returns:
            DataFrame with a new '{col}_anomaly' column
        """
        df = df.copy()
        if col not in df.columns or "date" not in df.columns:
            return df
        df["_doy"] = pd.to_datetime(df["date"]).dt.day_of_year
        doy_mean = df.groupby("_doy")[col].transform("mean")
        df[f"{col}_anomaly"] = df[col] - doy_mean
        df.drop(columns=["_doy"], inplace=True)
        return df

    def add_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sin/cos Fourier encoding of day-of-year and month.

        Fourier features give models a smooth, continuous representation of
        the annual cycle, avoiding the artificial discontinuity at year boundaries
        that a raw integer day-of-year feature would create.

        Args:
            df: Input DataFrame with a 'date' column

        Returns:
            DataFrame with doy_sin, doy_cos, month_sin, month_cos columns
        """
        df = df.copy()
        dates = pd.to_datetime(df["date"])
        doy = dates.dt.day_of_year
        month = dates.dt.month
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        return df

    def add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add integer day index as a linear trend feature.

        A trend feature lets models capture long-term changes such as
        gradual warming due to climate change over multi-year datasets.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with a 'trend' column (0, 1, 2, ...)
        """
        df = df.copy()
        df["trend"] = np.arange(len(df))
        return df

    def build_features(
        self,
        df: pd.DataFrame,
        target_col: str = "tmax_tomorrow",
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply all feature transforms and return (X, y) for model training.

        Creates a binary classification target: 1 if tomorrow's tmax exceeds
        threshold, 0 otherwise. If threshold is None, the median of tmax is used.

        Args:
            df: Raw weather DataFrame with at least 'date', 'tmax', 'tmin',
                'prcp', 'snowfall' columns.
            target_col: Name for the target column (used internally)
            threshold: Decision boundary for binary target. Defaults to median tmax.

        Returns:
            Tuple of (X DataFrame of features, y binary Series)
        """
        df = df.copy().reset_index(drop=True)

        # Build binary target: does tomorrow's tmax exceed threshold?
        if threshold is None:
            threshold = float(df["tmax"].median())
        df[target_col] = (df["tmax"].shift(-1) > threshold).astype(float)

        weather_cols = [c for c in ["tmax", "tmin", "prcp", "snowfall"] if c in df.columns]

        df = self.add_lags(df, cols=weather_cols, lags=[1, 2, 3, 7, 14])
        df = self.add_rolling_stats(df, cols=weather_cols, windows=[7, 14, 30])
        for col in weather_cols:
            df = self.add_anomalies(df, col=col)
        df = self.add_seasonality(df)
        df = self.add_trend(df)

        # Drop rows with NaN in any column (from lags / rolling)
        df = df.dropna().reset_index(drop=True)

        non_feature_cols = {"date", "station_id", target_col}
        feature_cols = [c for c in df.columns if c not in non_feature_cols]

        X = df[feature_cols]
        y = df[target_col]
        return X, y
