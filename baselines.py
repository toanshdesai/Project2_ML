"""
baselines.py - two "dumb" forecasts every ML model has to beat.

  Persistence: every future day's high = today's high.
               Hard to beat at day 1, useless by day 7.
  Climatology: the average high for that calendar date in the training years.
               Knows nothing about this week, but never drifts off.

A model error number means nothing on its own. "My model is off by 4.6 deg F" only
matters if climatology is off by 6.5 deg F. Both classes use the same fit/predict
interface as the real model, so evaluate.py treats all of them identically.

AI use: this file was written with help from Claude (Anthropic's AI assistant).
See the "AI use" section of README.md.
"""
import pandas as pd

import config


class Persistence:
    name = "Persistence"

    def fit(self, X, Y):
        self.columns = list(Y.columns)
        return self

    def predict(self, X):
        return pd.DataFrame({c: X["tmax"].to_numpy() for c in self.columns}, index=X.index)


class Climatology:
    name = "Climatology"

    def __init__(self, smooth_days=31):
        self.smooth_days = smooth_days   # smooth the day-of-year averages so one odd year doesn't create spikes

    def fit(self, X, Y):
        self.columns = list(Y.columns)
        self.horizons = [int(c.split("_d")[-1]) for c in self.columns]

        # Average high for each day of the year (1..366), using TRAINING rows only.
        by_doy = X["tmax"].groupby(X.index.dayofyear).mean().reindex(range(1, 367))
        by_doy = by_doy.interpolate(limit_direction="both")

        # Circular rolling mean: pad with the year's end/start so January smooths using December.
        pad = self.smooth_days // 2
        padded = pd.concat([by_doy.iloc[-pad:], by_doy, by_doy.iloc[:pad]])
        smooth = padded.rolling(self.smooth_days, center=True, min_periods=1).mean()
        self.normal_high = pd.Series(smooth.iloc[pad:-pad].to_numpy(), index=range(1, 367))
        return self

    def predict(self, X):
        out = {}
        for col, h in zip(self.columns, self.horizons):
            target_doy = (X.index + pd.Timedelta(days=h)).dayofyear
            out[col] = self.normal_high.loc[target_doy].to_numpy()
        return pd.DataFrame(out, index=X.index)