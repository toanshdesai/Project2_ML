# Project2_ML

This project predicts the daily high temperature in Durham, NC for each of the next 7 days.
It is written in Python and uses scikit-learn for the models, pandas for the data, and matplotlib for the charts.

The data comes from the Open-Meteo weather API (free, no key needed): 26 years of daily Durham weather, about 9,800 days.
The models train on 2000–2021 and are tested on 2022–2026, which they never see during training.

## How gradient boosting works

prediction = average + 0.03 × tree₁ + 0.03 × tree₂ + ... + 0.03 × tree₂₀₀

A decision tree is a flowchart of yes/no questions, like "is today's high above 72?" and then "is the pressure falling?",
and each end of the flowchart guesses a temperature.

Gradient boosting starts with the average high as its guess. It looks at how wrong that guess is on every training day,
then builds a small tree whose only job is to predict those misses. It adds 3% of that tree's correction (the learning rate)
and repeats with the new misses, 200 times. The final prediction is the average plus all 200 small corrections.

Ridge regression is one formula, `w1 × today's high + w2 × pressure + ...`, like a line of best fit with 24 inputs.
Each input always pushes the guess the same way, so it can't learn something like "a pressure drop matters more in winter."
Gradient boosting can, because one question in the tree can come after another.

The two baselines don't learn anything. **Persistence** guesses every day next week will be as hot as today.
**Climatology** guesses the normal high for that date, averaged over the training years.

## Design notes

- The data is split by date, not randomly. Two days in a row are almost the same, so if June 15 is in training and June 16 is in testing,
the model already knows the answer and the score is fake.
- Features only look backward (`shift(k)`) and the answers only look forward (`shift(-h)`), so the model never sees the future.
- There are 24 features: today's weather, the high 1, 2, 3 and 7 days ago, 3- and 7-day averages, and pressure change,
since a falling barometer usually means a front is coming.
- The day of the year is turned into a point on a circle (sin and cos), so Dec 31 and Jan 1 are next to each other instead of 364 days apart.
Wind direction gets the same treatment so 359° and 1° are both north.
- Each model is really 7 models, one per lead day. Predicting tomorrow and feeding that guess back in for the next day would let mistakes pile up.
- Gradient boosting was tuned with a grid search of 16 settings using `TimeSeriesSplit`, which always checks on years after the ones it trained on.
The test years are never used for tuning. The best settings are saved in `models/best_gbm_params.json`.
- Error is measured in degrees (MAE, the average of how far off each guess was). Accuracy/precision/recall are for yes/no predictions,
and this predicts a number.

## Files

- `main.py` — runs everything: train/evaluate, `--predict`, `--score`, `--tune`
- `config.py` — all settings: location, dates, train/test split, file paths
- `data_fetcher.py` — downloads weather from the Open-Meteo API and caches it in `data/durham_daily.csv`
- `features.py` — builds the 24 features and the 7 answers (the high 1–7 days later)
- `baselines.py` — persistence and climatology
- `model.py` — Ridge, gradient boosting, the 7-model wrapper, and tuning
- `evaluate.py` — error tables and the charts in `plots/`

## To use

Run with Python 3.10+ after installing the libraries (`pip install -r requirements.txt`).

- `python main.py` — load the data, train all four forecasters, print the results, and save the charts
- `python main.py --predict` — live 7-day forecast for Durham, next to the official forecast and the normal high
- `python main.py --score` — a few days after `--predict`, fills in the actual highs and shows who was closer
- `python main.py --tune` — rerun the grid search (takes a few minutes)
- `python main.py --refresh` — re-download all the weather data

The first run downloads the data (about a minute). After that it only downloads new days. Training takes about a minute.
In PyCharm, flags go in Run → Edit Configurations → Script parameters.

## Test Results

Average error in °F on the test years (2022–2026), lower is better.

| Days ahead        | 1    | 2    | 3    | 4    | 5    | 6    | 7    |
|-------------------|------|------|------|------|------|------|------|
| Persistence       | 5.14 | 7.16 | 7.85 | 8.39 | 8.70 | 8.64 | 8.64 |
| Climatology       | 6.45 | 6.44 | 6.44 | 6.44 | 6.44 | 6.44 | 6.44 |
| Ridge             | 4.35 | 5.71 | 6.19 | 6.36 | 6.41 | 6.45 | 6.48 |
| Gradient Boosting | 4.10 | 5.57 | 6.06 | 6.28 | 6.40 | 6.46 | 6.52 |

Gradient boosting was best for days 1–5. For tomorrow it was 36% more accurate than guessing the normal temperature.
Persistence beat climatology for tomorrow but was the worst by day 2. After about 4 days both ML models are no better than climatology.

The feature importance chart (`plots/feature_importance.png`) shows why. For tomorrow the model relies mostly on today's high,
but for 7 days out it relies almost only on the day of the year. It figured out on its own that today's weather says
almost nothing about next week, so it falls back on the normal temperature.

Unexpected: at days 6 and 7 the ML models are slightly worse than climatology (by less than 0.1°).
I think there's no real pattern left that far out, so the model is picking up a little random noise from the training years.

Live test: I ran `--predict` on Oct 13, 2026, and Durham had a heat wave, almost 20° above normal. My model missed it
and the official forecast got it almost exactly.

| Day           | Wed  | Thu  | Fri  | Sat  | Sun  |
|---------------|------|------|------|------|------|
| Actual high   | 90.8 | 88.6 | 87.7 | 73.5 | 60.4 |
| My model      | 82.4 | 75.9 | 71.8 | 68.0 | 69.0 |
| Official      | 91.1 | 89.0 | 87.8 | 69.9 | 64.9 |

My model only sees weather in Durham, so it can't see warm air coming from somewhere else.
The official forecast simulates the whole atmosphere, so it can.

## Limitations

- One location only. Adding weather from cities to the west and south (Charlotte, Atlanta) could let the model see weather systems coming.
- Training uses the ERA5 archive data, but `--predict` uses the forecast service's recent data. They are close but not identical.
- Only predicts the high. The same setup could predict the low, or rain/no rain (which would be a classification problem).
- The live test is only one week of forecasts, so it's an example, not a full test.

## AI use
Used Claude chats and Claude Code, did not use FlintK12

**Claude:** https://claude.ai/share/ab573c11-edd3-4431-926f-96460099e8f8