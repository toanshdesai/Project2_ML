"""
data_fetcher.py - pulls weather data from the Open-Meteo API (no API key needed)
and returns one tidy row per day.

Two endpoints:
  ARCHIVE_URL   ERA5 reanalysis, 1940 -> ~5 days ago. Used for TRAINING.
  FORECAST_URL  Live service. `past_days` gives recent days (inputs for a live
                prediction); future days are the professional forecast we compare against.

Data source: Open-Meteo.com, licensed CC BY 4.0.
"""
import time
from datetime import date, timedelta

import pandas as pd
import requests

import config

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Daily variables the API computes for us (names are from the Open-Meteo docs).
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
]

# Hourly variables we average into daily means ourselves. Pressure, dew point,
# clouds and humidity are only guaranteed as HOURLY fields on both endpoints,
# so we aggregate them in pandas instead of trusting a daily version exists.
HOURLY_VARS = [
    "pressure_msl",
    "dew_point_2m",
    "cloud_cover",
    "relative_humidity_2m",
]

# Short column names used everywhere else in the project.
RENAME = {
    "temperature_2m_max": "tmax",
    "temperature_2m_min": "tmin",
    "precipitation_sum": "precip",
    "wind_speed_10m_max": "wind_max",
    "wind_direction_10m_dominant": "wind_dir",
    "pressure_msl": "pressure",
    "dew_point_2m": "dewpoint",
    "cloud_cover": "cloud",
    "relative_humidity_2m": "humidity",
}

CHUNK_YEARS = 5   # split long history downloads so no single response is huge


class WeatherAPI:
    """Thin wrapper around Open-Meteo. Every public method returns a DataFrame indexed by date."""

    def __init__(self, lat=config.LAT, lon=config.LON, timezone=config.TIMEZONE):
        self.lat = lat
        self.lon = lon
        self.timezone = timezone

    # ------------------------------------------------------------------ helpers
    def _base_params(self):
        return {
            "latitude": self.lat,
            "longitude": self.lon,
            "timezone": self.timezone,          # required whenever daily variables are requested
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
        }

    def _request(self, url, params, retries=3):
        """GET with a timeout and simple retry/backoff, so one network hiccup doesn't kill the run."""
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, timeout=90)
                response.raise_for_status()
                payload = response.json()
                if payload.get("error"):
                    raise RuntimeError(f"Open-Meteo error: {payload.get('reason')}")
                return payload
            except (requests.RequestException, RuntimeError) as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"  request failed ({exc}); retrying in {wait}s")
                time.sleep(wait)

    @staticmethod
    def _to_daily_frame(payload):
        """Turn the API's JSON (parallel lists) into one row per day."""
        daily = pd.DataFrame(payload["daily"])
        daily["time"] = pd.to_datetime(daily["time"])
        daily = daily.set_index("time")

        hourly = pd.DataFrame(payload["hourly"])
        hourly["time"] = pd.to_datetime(hourly["time"])
        hourly_means = hourly.set_index("time").resample("D").mean()   # 24 hourly values -> 1 daily mean

        df = daily.join(hourly_means, how="left").rename(columns=RENAME)
        df.index.name = "date"
        return df

    # ------------------------------------------------------------- public API
    def fetch_history(self, start=config.HISTORY_START, end=None, refresh=False):
        """
        Daily history from the ERA5 archive, cached to data/durham_daily.csv.

        First run downloads everything. Later runs load the CSV and only download
        days added since the last run. refresh=True forces a full re-download.
        """
        if end is None:
            end = (date.today() - timedelta(days=config.ARCHIVE_DELAY_DAYS)).isoformat()
        end_ts = pd.Timestamp(end)

        cached = None
        if config.HISTORY_CSV.exists() and not refresh:
            cached = pd.read_csv(config.HISTORY_CSV, index_col="date", parse_dates=True)
            if cached.index.max() >= end_ts:
                print(f"Loaded {len(cached):,} cached days ({cached.index.min().date()} -> {cached.index.max().date()})")
                return cached.loc[:end_ts]
            start = (cached.index.max() + pd.Timedelta(days=1)).date().isoformat()
            print(f"Cache ends {cached.index.max().date()}; downloading {start} -> {end}")

        frames = []
        chunk_start = pd.Timestamp(start)
        while chunk_start <= end_ts:
            chunk_end = min(chunk_start + pd.DateOffset(years=CHUNK_YEARS) - pd.Timedelta(days=1), end_ts)
            print(f"  downloading {chunk_start.date()} -> {chunk_end.date()}")
            params = self._base_params() | {
                "start_date": chunk_start.date().isoformat(),
                "end_date": chunk_end.date().isoformat(),
                "daily": ",".join(DAILY_VARS),
                "hourly": ",".join(HOURLY_VARS),
            }
            frames.append(self._to_daily_frame(self._request(ARCHIVE_URL, params)))
            chunk_start = chunk_end + pd.Timedelta(days=1)

        new = pd.concat(frames)
        df = new if cached is None else pd.concat([cached, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_csv(config.HISTORY_CSV)
        print(f"Saved {len(df):,} days to {config.HISTORY_CSV.name}")
        return df

    def fetch_recent(self, past_days=30):
        """
        The last `past_days` COMPLETE days from the live endpoint (today is excluded
        because it isn't over yet). These rows feed the model for a live prediction.
        """
        params = self._base_params() | {
            "past_days": past_days,
            "forecast_days": 1,
            "daily": ",".join(DAILY_VARS),
            "hourly": ",".join(HOURLY_VARS),
        }
        df = self._to_daily_frame(self._request(FORECAST_URL, params))
        return df[df.index < pd.Timestamp(date.today())]

    def fetch_official_forecast(self, days=8):
        """The professional forecast's daily high for today + the next days-1 days."""
        params = self._base_params() | {
            "forecast_days": days,
            "daily": "temperature_2m_max",
        }
        payload = self._request(FORECAST_URL, params)
        daily = pd.DataFrame(payload["daily"])
        return pd.Series(daily["temperature_2m_max"].values,
                         index=pd.to_datetime(daily["time"]), name="official")