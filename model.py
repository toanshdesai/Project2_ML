"""
model.py - the actual ML.

WeekForecaster trains SEVEN separate regressors, one per lead day ("direct
multi-horizon" forecasting). The alternative, predicting day 1 and feeding that
prediction back in to get day 2 ("recursive"), compounds its own errors.
Seven models cost seven times the training, but each one is honest about its horizon.

Two model families:
  Ridge         linear regression with a penalty on large weights. Fast, explainable baseline ML.
  GradientBoost HistGradientBoostingRegressor: hundreds of small decision trees, each one
                correcting the last. Captures non-linear effects (e.g. a pressure drop matters
                more in winter).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config


# ---------------------------------------------------------------- factories
# Each factory returns a FRESH, untrained estimator, so each of the 7 horizons gets its own.

def make_ridge():
    # Scale features first: Ridge's penalty is unfair to features measured in big units (pressure ~1015 hPa).
    # RidgeCV picks the penalty strength alpha using time-ordered cross-validation.
    return make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=np.logspace(-2, 3, 12), cv=TimeSeriesSplit(n_splits=5)),
    )


def make_gbm(**params):
    defaults = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                    min_samples_leaf=30, l2_regularization=1.0,
                    random_state=config.RANDOM_STATE)
    defaults.update(params)
    return HistGradientBoostingRegressor(**defaults)


# ------------------------------------------------------------------- tuning
def tune_gbm(X_train, y_train):
    """
    Grid search with TimeSeriesSplit: every validation fold comes AFTER its training fold,
    the same way the model is used in real life. Plain k-fold would let the model train on
    2015 and validate on 2010, which leaks seasonal patterns from the future.
    """
    grid = {
        "learning_rate": [0.03, 0.1],
        "max_leaf_nodes": [15, 31],
        "max_iter": [200, 400],
        "min_samples_leaf": [20, 60],
    }
    search = GridSearchCV(
        make_gbm(),
        grid,
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    results = (pd.DataFrame(search.cv_results_)
               .sort_values("rank_test_score")
               [["params", "mean_test_score", "std_test_score"]])
    results["cv_mae"] = -results.pop("mean_test_score")
    return search.best_params_, results


# ------------------------------------------------------------------ model
class WeekForecaster:
    def __init__(self, make_estimator, name):
        self.make_estimator = make_estimator
        self.name = name
        self.models = {}          # e.g. {"tmax_d1": fitted_estimator, ...}
        self.features = None

    def fit(self, X, Y):
        self.features = list(X.columns)
        for col in Y.columns:
            est = self.make_estimator()
            est.fit(X, Y[col])
            self.models[col] = est
        return self

    def predict(self, X):
        X = X[self.features]      # same columns, same order as training
        return pd.DataFrame({col: est.predict(X) for col, est in self.models.items()}, index=X.index)

    def save(self, path):
        joblib.dump({"name": self.name, "models": self.models, "features": self.features}, path)

    @classmethod
    def load(cls, path):
        blob = joblib.load(path)
        obj = cls(make_estimator=None, name=blob["name"])
        obj.models, obj.features = blob["models"], blob["features"]
        return obj