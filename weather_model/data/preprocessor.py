"""Data cleaning, normalization, and splitting utilities."""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class DataPreprocessor:
    """Cleans, normalizes, and splits weather DataFrames.

    Stores normalization parameters so inverse transform is possible at inference.
    Uses IQR-based outlier removal to handle sensor errors and transcription mistakes.
    Temporal splitting prevents data leakage that would inflate backtest performance.

    Attributes:
        _means: dict of column means from fit
        _stds: dict of column stds from fit
        numeric_cols: list of numeric columns to normalize
    """

    def __init__(self, numeric_cols: Optional[list] = None):
        self.numeric_cols = numeric_cols or ["tmax", "tmin", "prcp", "snowfall"]
        self._means: dict = {}
        self._stds: dict = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method and fill missing values.

        Outliers beyond 3 IQR from Q1/Q3 are replaced with NaN, then
        forward-filled (using prior observation) to preserve time series continuity.

        Args:
            df: Raw weather DataFrame

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
            df.loc[(df[col] < lower) | (df[col] > upper), col] = np.nan
        df[self.numeric_cols] = df[self.numeric_cols].ffill().bfill()
        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Z-score normalize numeric columns.

        Args:
            df: DataFrame to normalize
            fit: If True, compute and store mean/std from this data.
                 If False, use stored params.

        Returns:
            Normalized DataFrame
        """
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            if fit:
                self._means[col] = df[col].mean()
                self._stds[col] = df[col].std() or 1.0
            mean = self._means.get(col, 0.0)
            std = self._stds.get(col, 1.0)
            df[col] = (df[col] - mean) / std
        return df

    def inverse_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse z-score normalization using stored parameters.

        Args:
            df: Normalized DataFrame

        Returns:
            DataFrame in original units
        """
        df = df.copy()
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            mean = self._means.get(col, 0.0)
            std = self._stds.get(col, 1.0)
            df[col] = df[col] * std + mean
        return df

    def split_train_test(
        self, df: pd.DataFrame, test_frac: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Temporal train/test split to prevent data leakage.

        Always uses the last test_frac portion of the time series as the test set.
        This mirrors real-world deployment where we always predict future data.

        Args:
            df: Full time series DataFrame sorted by date
            test_frac: Fraction of data for test set

        Returns:
            Tuple of (train_df, test_df)
        """
        n = len(df)
        cutoff = int(n * (1 - test_frac))
        return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
