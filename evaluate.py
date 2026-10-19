"""
evaluate.py - scores every forecaster on the held-out test years and draws the charts.

Metrics
  MAE   mean absolute error in deg F: "on average, how many degrees off?" Easy to explain
        to a non-technical audience, and less dominated by one freak day than RMSE.
  RMSE  root mean squared error: punishes big misses harder. Reported alongside MAE.
  Skill 1 - MAE_model / MAE_climatology. 0 = no better than the calendar average,
        0.30 = 30% less error than climatology, negative = worse than just guessing normal.

AI use: this file was written with help from Claude (Anthropic's AI assistant).
See the "AI use" section of README.md.
"""
import matplotlib
matplotlib.use("Agg")    # render to files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config

# Fixed colors per forecaster: the same model is the same color in every chart.
COLORS = {
    "Gradient Boosting": "#2a78d6",
    "Ridge": "#eb6834",
    "Persistence": "#1baf7a",
    "Climatology": "#eda100",
}
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 13, "axes.titleweight": "bold", "font.size": 10,
    "legend.frameon": False,
})


class Evaluator:
    def __init__(self, Y_true):
        self.Y_true = Y_true
        self.leads = [int(c.split("_d")[-1]) for c in Y_true.columns]

    # ------------------------------------------------------------- numbers
    def error_table(self, predictions, metric="mae"):
        """Rows = forecaster, columns = lead day 1..7."""
        rows = {}
        for name, pred in predictions.items():
            err = pred[self.Y_true.columns].to_numpy() - self.Y_true.to_numpy()
            values = np.abs(err).mean(axis=0) if metric == "mae" else np.sqrt((err ** 2).mean(axis=0))
            rows[name] = values
        return pd.DataFrame(rows, index=self.leads).T.rename_axis("lead_day", axis=1)

    @staticmethod
    def skill_table(mae, reference="Climatology"):
        return 1 - mae / mae.loc[reference]

    # -------------------------------------------------------------- charts
    def plot_error_by_lead(self, mae, path):
        """The headline chart: how fast does each forecaster's error grow with lead time?"""
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for name in mae.index:
            ax.plot(self.leads, mae.loc[name], color=COLORS.get(name, INK), lw=2,
                    marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        # Direct labels at the line ends, nudged apart so close lines don't print on top of each other.
        ends = mae[self.leads[-1]].sort_values()
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        label_y, last = {}, -np.inf
        for name, y in ends.items():
            y_lab = max(y, last + 0.05 * span)
            label_y[name], last = y_lab, y_lab
        for name, y in ends.items():
            ax.annotate(f"{name}  {y:.2f}", (self.leads[-1], y), xytext=(self.leads[-1] + 0.2, label_y[name]),
                        textcoords="data", va="center", color=INK_2, fontsize=9)
        ax.set_xticks(self.leads)
        ax.set_xlim(0.7, self.leads[-1] + 1.8)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Days ahead")
        ax.set_ylabel("Mean absolute error (deg F)")
        ax.set_title(f"Forecast error by lead day, Durham daily high ({self._test_years()})", loc="left")
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def _test_years(self):
        return f"test {self.Y_true.index[0].year}-{self.Y_true.index[-1].year}"

    def plot_skill_by_lead(self, skill, path, models=("Gradient Boosting", "Ridge")):
        """
        Zoomed-in view of the same results: skill vs climatology, where the zero line IS climatology.
        Above zero = the model beats the seasonal normal; at or below zero = no useful skill left.
        """
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.axhline(0, color=MUTED, lw=1.5, ls="--", zorder=1)
        ax.annotate("Climatology = 0 (no skill)", (self.leads[0], 0), xytext=(self.leads[0] - 0.2, -0.012),
                    textcoords="data", ha="left", va="top", color=INK_2, fontsize=9)

        for name in models:
            values = skill.loc[name]
            ax.plot(self.leads, values, color=COLORS.get(name, INK), lw=2, marker="o", ms=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
        # Label every point of the best model so the decay is readable without the table.
        best = models[0]
        for lead, v in zip(self.leads, skill.loc[best]):
            text = f"{v:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")
            ax.annotate(text, (lead, v), xytext=(8, 8), textcoords="offset points",
                        ha="left", va="bottom", color=INK_2, fontsize=9)

        top = max(skill.loc[list(models)].to_numpy().max(), 0.05)
        ax.set_ylim(-0.08, top + 0.08)
        ax.set_xticks(self.leads)
        ax.set_xlim(0.7, self.leads[-1] + 1.8)
        ax.set_xlabel("Days ahead")
        ax.set_ylabel("Skill = 1 - MAE / MAE of climatology")
        ax.set_title(f"How long does the model beat the seasonal normal? ({self._test_years()})", loc="left")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def plot_forecast_vs_actual(self, predictions, model_name, path, start=None, days=90):
        """One stretch of the test period: what actually happened vs the 1-day and 7-day forecasts."""
        pred = predictions[model_name]
        start = pd.Timestamp(start) if start else self.Y_true.index[0]
        window = self.Y_true.loc[start:].index[:days]
        actual = self.Y_true.loc[window, "tmax_d1"]
        actual.index = actual.index + pd.Timedelta(days=1)          # tmax_d1 at issue date t happened on t+1

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(actual.index, actual, color=INK, lw=2, label="Actual high")
        for lead, style in [(1, "-"), (7, "--")]:                  # same model = same color; lead day = line style
            s = pred.loc[window, f"tmax_d{lead}"]
            s.index = s.index + pd.Timedelta(days=lead)              # plot each forecast on the day it was FOR
            s = s.loc[actual.index.min():actual.index.max()]         # same date range as the actual line
            ax.plot(s.index, s, color=COLORS.get(model_name, INK_2), lw=2, ls=style,
                    label=f"{model_name}, {lead}-day-ahead forecast")
        ax.set_ylabel("Daily high (deg F)")
        ax.set_title(f"Actual vs forecast, {window[0]:%b %Y} - {window[-1]:%b %Y}", loc="left")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def plot_importance(self, forecaster, X_test, path, leads=(1, 7), top=12):
        """
        Permutation importance: shuffle one feature, measure how much worse MAE gets.
        Works for any model and is measured on TEST data, so it reflects what the model actually relies on.
        """
        fig, axes = plt.subplots(1, len(leads), figsize=(12, 5.5), sharex=False)
        for ax, lead in zip(np.atleast_1d(axes), leads):
            col = f"tmax_d{lead}"
            result = permutation_importance(
                forecaster.models[col], X_test[forecaster.features], self.Y_true.loc[X_test.index, col],
                scoring="neg_mean_absolute_error", n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1)
            imp = pd.Series(result.importances_mean, index=forecaster.features).sort_values().tail(top)
            ax.barh(imp.index, imp.values, color=COLORS.get(forecaster.name, INK), height=0.7)
            ax.set_title(f"{lead}-day-ahead", loc="left")
            ax.set_xlabel("MAE increase when shuffled (deg F)")
            ax.grid(axis="y", visible=False)
        fig.suptitle(f"What the {forecaster.name} model relies on", x=0.01, ha="left",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)