"""
config.py - every setting the project uses, in one place.

Change the location or the date split here and every other file picks it up.

AI use: this file was written with help from Claude (Anthropic's AI assistant).
See the "AI use" section of README.md.
"""
from pathlib import Path

# ---- Location: Durham, NC -------------------------------------------------
LAT = 35.99
LON = -78.90
TIMEZONE = "America/New_York"   # daily boundaries (midnight to midnight) in local time

# ---- Data range -----------------------------------------------------------
HISTORY_START = "2000-01-01"    # ~26 years of daily rows is plenty for tree models
ARCHIVE_DELAY_DAYS = 7          # ERA5 archive runs ~5 days behind real time; 7 is a safe margin

# ---- Train / test split (by DATE, never shuffled) -------------------------
# Everything up to TRAIN_END is used to fit models. Everything from TEST_START on
# is held out and only touched once, for the final scores. Shuffling would leak
# tomorrow's weather into training, which inflates scores and is the #1 mistake
# in time-series ML.
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

# ---- Forecast setup -------------------------------------------------------
HORIZONS = list(range(1, 8))    # predict the daily high 1..7 days ahead
TARGET = "tmax"                 # the column we forecast (daily high, deg F)

# ---- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"

HISTORY_CSV = DATA_DIR / "durham_daily.csv"
RESULTS_CSV = PLOT_DIR / "results_mae.csv"
LIVE_LOG_CSV = LOG_DIR / "live_predictions.csv"

for _d in (DATA_DIR, PLOT_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

RANDOM_STATE = 42
