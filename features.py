"""
features.py - turns a daily weather table into a supervised-learning table.

Every row is a "forecast issue date" t.
  X (features) = only things known by the END of day t (today and the past).
  Y (targets)  = the daily high on day t+1, t+2, ... t+7.

The rule that keeps this honest: features may look BACKWARD (shift(+k)),
targets look FORWARD (shift(-h)). If a feature ever uses shift(-k), the model
is peeking at the future, and the test score is fiction.
"""
import numpy as np
import pandas as pd

import config


class FeatureBuilder:
    def __init__(self, horizons=config.HORIZONS, target=config.TARGET):
        self.horizons = horizons
        self.target = target
        self.feature_columns = None   # filled in by add_features()

    @staticmethod
    def _clean(df):
        """Fill short gaps (<= 3 days) by interpolation. ERA5 is gap-free; live data occasionally isn't."""
        df = df.sort_index().asfreq("D")            # guarantee one row per calendar day
        return df.interpolate(limit=3, limit_direction="forward")

    def add_features(self, df):
        """Return a DataFrame of features (same date index). Early rows contain NaN from lags."""
        df = self._clean(df)
        f = pd.DataFrame(index=df.index)

        # 1) Today's conditions
        for col in ["tmax", "tmin", "precip", "pressure", "dewpoint", "cloud", "humidity", "wind_max"]:
            f[col] = df[col]

        # Wind direction is circular (359 deg and 1 deg are neighbors), so encode it as a point on a circle.
        rad = np.deg2rad(df["wind_dir"])
        f["wind_dir_sin"] = np.sin(rad)
        f["wind_dir_cos"] = np.cos(rad)

        # 2) Lags - what the high was 1, 2, 3 and 7 days ago
        for k in [1, 2, 3, 7]:
            f[f"tmax_lag{k}"] = df["tmax"].shift(k)
        f["tmin_lag1"] = df["tmin"].shift(1)

        # 3) Trends - recent averages and changes (a falling barometer means a front is coming)
        f["tmax_roll3"] = df["tmax"].rolling(3).mean()
        f["tmax_roll7"] = df["tmax"].rolling(7).mean()
        f["tmax_change1"] = df["tmax"] - df["tmax"].shift(1)
        f["pressure_change1"] = df["pressure"] - df["pressure"].shift(1)
        f["pressure_change3"] = df["pressure"] - df["pressure"].shift(3)
        f["dewpoint_change1"] = df["dewpoint"] - df["dewpoint"].shift(1)
        f["precip_sum3"] = df["precip"].rolling(3).sum()

        # 4) Season - day of year as a point on a circle, so Dec 31 sits next to Jan 1
        doy = df.index.dayofyear
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

        self.feature_columns = list(f.columns)
        return f

    def make_targets(self, df):
        """Y[t, 'tmax_d{h}'] = daily high on day t+h."""
        df = self._clean(df)
        return pd.DataFrame(
            {f"{self.target}_d{h}": df[self.target].shift(-h) for h in self.horizons},
            index=df.index,
        )

    def build_training_table(self, df):
        """Aligned X and Y with every incomplete row dropped (first 7 days: no lags; last 7: no targets)."""
        X = self.add_features(df)
        Y = self.make_targets(df)
        keep = X.notna().all(axis=1) & Y.notna().all(axis=1)
        return X[keep], Y[keep]
