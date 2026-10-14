"""
main.py - runs the whole pipeline.

  python main.py             download/load history -> train -> evaluate -> charts
  python main.py --tune      same, but grid-search the gradient boosting settings first
  python main.py --refresh   force a full re-download of the history
  python main.py --predict   live 7-day forecast for Durham vs the official forecast
"""
import argparse
import json
from datetime import date

import joblib
import pandas as pd

import config
from baselines import Climatology, Persistence
from data_fetcher import WeatherAPI
from evaluate import Evaluator
from features import FeatureBuilder
from model import WeekForecaster, make_gbm, make_ridge, tune_gbm

GBM_PATH = config.MODEL_DIR / "gradient_boosting.joblib"
CLIM_PATH = config.MODEL_DIR / "climatology.joblib"
PARAMS_PATH = config.MODEL_DIR / "best_gbm_params.json"


def split_by_date(X, Y):
    train = X.index <= pd.Timestamp(config.TRAIN_END)
    test = X.index >= pd.Timestamp(config.TEST_START)
    return X[train], Y[train], X[test], Y[test]


def train_and_evaluate(refresh=False, tune=False):
    # ---- 1. Data
    api = WeatherAPI()
    history = api.fetch_history(refresh=refresh)

    # ---- 2. Features + targets, split by date
    fb = FeatureBuilder()
    X, Y = fb.build_training_table(history)
    X_train, Y_train, X_test, Y_test = split_by_date(X, Y)
    print(f"\nTrain: {len(X_train):,} days ({X_train.index.min().date()} -> {X_train.index.max().date()})")
    print(f"Test:  {len(X_test):,} days ({X_test.index.min().date()} -> {X_test.index.max().date()})")
    print(f"Features ({len(fb.feature_columns)}): {', '.join(fb.feature_columns)}")

    # ---- 3. Optional tuning (training data only; test set stays untouched)
    gbm_params = {}
    if tune:
        print("\nTuning gradient boosting with TimeSeriesSplit (lead day 1)...")
        gbm_params, cv_results = tune_gbm(X_train, Y_train["tmax_d1"])
        cv_results.to_csv(config.PLOT_DIR / "tuning_results.csv", index=False)
        PARAMS_PATH.write_text(json.dumps(gbm_params, indent=2))
        print(f"Best params: {gbm_params}  (CV MAE {cv_results['cv_mae'].iloc[0]:.2f} deg F)")
    elif PARAMS_PATH.exists():
        gbm_params = json.loads(PARAMS_PATH.read_text())
        print(f"\nUsing saved tuned params: {gbm_params}")

    # ---- 4. Fit every forecaster on the SAME training rows
    forecasters = {
        "Persistence": Persistence(),
        "Climatology": Climatology(),
        "Ridge": WeekForecaster(make_ridge, "Ridge"),
        "Gradient Boosting": WeekForecaster(lambda: make_gbm(**gbm_params), "Gradient Boosting"),
    }
    print("\nTraining...")
    for name, f in forecasters.items():
        f.fit(X_train, Y_train)
        print(f"  {name} done")

    # ---- 5. Score on the held-out test years
    predictions = {name: f.predict(X_test) for name, f in forecasters.items()}
    ev = Evaluator(Y_test)
    mae = ev.error_table(predictions, "mae")
    rmse = ev.error_table(predictions, "rmse")
    skill = ev.skill_table(mae)

    pd.set_option("display.float_format", "{:.2f}".format)
    print("\nMAE (deg F) by lead day, lower is better:\n", mae)
    print("\nRMSE (deg F) by lead day:\n", rmse)
    print("\nSkill vs climatology (1 - MAE/MAE_clim), higher is better, 0 = no skill:\n", skill)

    pd.concat({"MAE": mae, "RMSE": rmse, "Skill": skill}).to_csv(config.RESULTS_CSV)

    # ---- 6. Charts
    ev.plot_error_by_lead(mae, config.PLOT_DIR / "error_by_lead_day.png")
    ev.plot_skill_by_lead(skill, config.PLOT_DIR / "skill_by_lead_day.png")
    ev.plot_forecast_vs_actual(predictions, "Gradient Boosting", config.PLOT_DIR / "forecast_vs_actual.png",
                               start=f"{X_test.index[0].year}-01-01", days=120)
    print("\nComputing feature importance (takes ~30s)...")
    ev.plot_importance(forecasters["Gradient Boosting"], X_test, config.PLOT_DIR / "feature_importance.png")
    print(f"Charts saved to {config.PLOT_DIR}")

    # ---- 7. Save the models for --predict
    # Refit on ALL rows (train + test) so the live model uses the most recent years too.
    final = WeekForecaster(lambda: make_gbm(**gbm_params), "Gradient Boosting").fit(X, Y)
    final.save(GBM_PATH)
    joblib.dump(Climatology().fit(X, Y), CLIM_PATH)
    print(f"Final model (trained on all {len(X):,} days) saved to {GBM_PATH.name}")


def predict_live():
    if not GBM_PATH.exists():
        print("No saved model yet - training first.")
        train_and_evaluate()

    model = WeekForecaster.load(GBM_PATH)
    clim = joblib.load(CLIM_PATH)

    api = WeatherAPI()
    recent = api.fetch_recent(past_days=30)
    feats = FeatureBuilder().add_features(recent).dropna()
    anchor = feats.index[-1]                       # most recent complete day = the forecast issue date
    ours = model.predict(feats.loc[[anchor]]).iloc[0]
    normal = clim.predict(feats.loc[[anchor]]).iloc[0]
    official = api.fetch_official_forecast(days=8)

    rows = []
    for h in config.HORIZONS:
        day = anchor + pd.Timedelta(days=h)
        rows.append({
            "issued": anchor.date(), "lead_day": h, "date": day.date(), "weekday": f"{day:%a}",
            "our_high": round(ours[f"tmax_d{h}"], 1),
            "official_high": round(official.get(day, float("nan")), 1),
            "normal_high": round(normal[f"tmax_d{h}"], 1),
        })
    table = pd.DataFrame(rows)

    print(f"\nDurham 7-day high forecast (inputs through {anchor:%a %b %d}, run {date.today()})\n")
    print(table[["date", "weekday", "our_high", "official_high", "normal_high"]].to_string(index=False))

    # Append to a running log so we can score live forecasts against what actually happens.
    table.assign(actual_high=pd.NA).to_csv(config.LIVE_LOG_CSV, mode="a",
                                           header=not config.LIVE_LOG_CSV.exists(), index=False)
    print(f"\nLogged to {config.LIVE_LOG_CSV.relative_to(config.ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Durham 7-day high-temperature forecaster")
    parser.add_argument("--predict", action="store_true", help="make a live 7-day forecast")
    parser.add_argument("--tune", action="store_true", help="grid-search gradient boosting settings first")
    parser.add_argument("--refresh", action="store_true", help="re-download the full history")
    args = parser.parse_args()

    if args.predict:
        predict_live()
    else:
        train_and_evaluate(refresh=args.refresh, tune=args.tune)
