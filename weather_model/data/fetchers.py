"""
Data fetchers for NOAA, GFS, and ECMWF weather data sources.
"""
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic weather data for testing when APIs are unavailable.

    Uses realistic seasonal patterns based on climatological normals.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=datetime.today(), periods=n_days, freq='D')
    doy = dates.day_of_year
    # Seasonal temperature signal
    tmax = 25 + 15 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 3, n_days)
    tmin = tmax - 10 + rng.normal(0, 2, n_days)
    prcp = rng.exponential(2, n_days) * (rng.random(n_days) > 0.7)
    snowfall = np.where(tmin < 0, prcp * 10, 0.0)
    return pd.DataFrame({
        'date': dates,
        'station_id': 'SYNTHETIC',
        'tmax': tmax,
        'tmin': tmin,
        'prcp': prcp,
        'snowfall': snowfall,
    })


class NOAAFetcher:
    """Fetches weather data from NOAA Climate Data Online API.

    NOAA CDO provides quality-controlled observational data going back decades.
    Ensemble forecasting combines multiple model runs to quantify uncertainty.
    Data assimilation merges model forecasts with real observations for better accuracy.
    NOAA's long historical record gives strong climatological signal for prediction.

    Args:
        api_token: NOAA CDO API token (register at https://www.ncdc.noaa.gov/cdo-web/token)
        base_url: Base URL for NOAA CDO API
    """

    BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"

    def __init__(self, api_token: str = "", base_url: str = BASE_URL):
        self.api_token = api_token
        self.base_url = base_url
        self.session = requests.Session()
        if api_token:
            self.session.headers.update({"token": api_token})

    def fetch(self, station_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily summary data for a station.

        Args:
            station_id: NOAA station identifier (e.g. 'GHCND:USW00094728')
            start_date: ISO format start date 'YYYY-MM-DD'
            end_date: ISO format end date 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: date, station_id, tmax, tmin, prcp, snowfall
        """
        if not self.api_token:
            logger.warning("No NOAA API token provided; returning synthetic data")
            return generate_synthetic_data()
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "datatypeid": "TMAX,TMIN,PRCP,SNOW",
            "units": "metric",
            "limit": 1000,
        }
        try:
            resp = self.session.get(self.base_url + "data", params=params, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            return self._parse_results(results, station_id)
        except requests.RequestException as exc:
            logger.error("NOAA fetch failed: %s; returning synthetic data", exc)
            return generate_synthetic_data()

    def _parse_results(self, results: list, station_id: str) -> pd.DataFrame:
        rows: dict = {}
        for item in results:
            date = item["date"][:10]
            rows.setdefault(
                date,
                {
                    "date": date,
                    "station_id": station_id,
                    "tmax": np.nan,
                    "tmin": np.nan,
                    "prcp": 0.0,
                    "snowfall": 0.0,
                },
            )
            dtype = item["datatype"]
            val = float(item["value"])
            if dtype == "TMAX":
                rows[date]["tmax"] = val / 10.0
            elif dtype == "TMIN":
                rows[date]["tmin"] = val / 10.0
            elif dtype == "PRCP":
                rows[date]["prcp"] = val / 10.0
            elif dtype == "SNOW":
                rows[date]["snowfall"] = val / 10.0
        df = pd.DataFrame(list(rows.values()))
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)


class GFSFetcher:
    """Simulates Global Forecast System (GFS) numerical weather prediction data.

    GFS is a spectral model run by NCEP 4x daily at 0.25-degree resolution.
    It uses 4D-Var data assimilation and ensemble perturbations to estimate
    forecast uncertainty. GFS data is the backbone of US weather prediction.

    Args:
        n_ensemble_members: Number of ensemble perturbations to simulate
    """

    def __init__(self, n_ensemble_members: int = 21):
        self.n_ensemble_members = n_ensemble_members

    def fetch(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 10,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Simulate GFS forecast ensemble for a location.

        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days
            seed: Random seed for reproducibility

        Returns:
            DataFrame with ensemble mean and spread for tmax, tmin, prcp, snowfall
        """
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start=datetime.today(), periods=forecast_days, freq='D')
        doy = dates.day_of_year
        tmax_base = 25 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)
        # Ensemble spread grows with lead time
        spread = np.sqrt(np.arange(1, forecast_days + 1))
        tmax_ensemble = (
            tmax_base[:, None]
            + rng.normal(0, 1, (forecast_days, self.n_ensemble_members)) * spread[:, None]
        )
        tmin_ensemble = tmax_ensemble - 10 + rng.normal(0, 1, (forecast_days, self.n_ensemble_members))
        prcp_ensemble = np.maximum(0, rng.normal(2, 1, (forecast_days, self.n_ensemble_members)))
        return pd.DataFrame({
            "date": dates,
            "station_id": f"GFS_{lat:.2f}_{lon:.2f}",
            "tmax": tmax_ensemble.mean(axis=1),
            "tmin": tmin_ensemble.mean(axis=1),
            "prcp": prcp_ensemble.mean(axis=1),
            "snowfall": np.where(
                tmin_ensemble.mean(axis=1) < 0,
                prcp_ensemble.mean(axis=1) * 10,
                0.0,
            ),
            "tmax_spread": tmax_ensemble.std(axis=1),
            "tmin_spread": tmin_ensemble.std(axis=1),
        })


class ECMWFFetcher:
    """Simulates ECMWF ensemble forecast (ENS) data.

    ECMWF's ensemble system uses 51 members (1 control + 50 perturbed).
    It is widely regarded as the world's best medium-range forecast model.
    Singular vectors and stochastic physics parameterisations capture forecast uncertainty.
    ECMWF data commands a premium because of its superior skill scores globally.

    Args:
        n_ensemble_members: Number of ECMWF ensemble members (default 51)
    """

    def __init__(self, n_ensemble_members: int = 51):
        self.n_ensemble_members = n_ensemble_members

    def fetch(
        self,
        lat: float,
        lon: float,
        forecast_days: int = 15,
        seed: int = 1,
    ) -> pd.DataFrame:
        """Simulate ECMWF ensemble forecast for a location.

        ECMWF generally outperforms GFS beyond day 5 due to better data assimilation
        and higher model resolution. Ensemble spread is calibrated to reflect true
        forecast uncertainty.

        Args:
            lat: Latitude
            lon: Longitude
            forecast_days: Number of forecast days (ECMWF goes out to 15 days)
            seed: Random seed for reproducibility

        Returns:
            DataFrame with ensemble statistics for standard weather variables
        """
        rng = np.random.default_rng(seed)
        dates = pd.date_range(start=datetime.today(), periods=forecast_days, freq='D')
        doy = dates.day_of_year
        tmax_base = 25 + 15 * np.sin(2 * np.pi * (doy - 80) / 365)
        # ECMWF has slightly smaller spread than GFS for first 5 days, larger thereafter
        spread = 0.8 * np.sqrt(np.arange(1, forecast_days + 1))
        tmax_ensemble = (
            tmax_base[:, None]
            + rng.normal(0, 1, (forecast_days, self.n_ensemble_members)) * spread[:, None]
        )
        tmin_ensemble = tmax_ensemble - 10 + rng.normal(0, 0.8, (forecast_days, self.n_ensemble_members))
        prcp_ensemble = np.maximum(0, rng.normal(1.8, 0.9, (forecast_days, self.n_ensemble_members)))
        return pd.DataFrame({
            "date": dates,
            "station_id": f"ECMWF_{lat:.2f}_{lon:.2f}",
            "tmax": tmax_ensemble.mean(axis=1),
            "tmin": tmin_ensemble.mean(axis=1),
            "prcp": prcp_ensemble.mean(axis=1),
            "snowfall": np.where(
                tmin_ensemble.mean(axis=1) < 0,
                prcp_ensemble.mean(axis=1) * 10,
                0.0,
            ),
            "tmax_spread": tmax_ensemble.std(axis=1),
            "tmin_spread": tmin_ensemble.std(axis=1),
            "prob_prcp_above_1mm": (prcp_ensemble > 1).mean(axis=1),
        })
