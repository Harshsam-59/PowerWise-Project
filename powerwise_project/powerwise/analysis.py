"""
analysis.py — Exploratory Data Analysis and Visualization
==========================================================
Responsibilities
- Compute descriptive statistics.
- Build Matplotlib / Seaborn figures for the Streamlit UI.
- Compute moving averages and monthly aggregations.

Design notes
- All plot functions return a Matplotlib Figure object.  Streamlit displays
  it with st.pyplot(fig).  This keeps the plotting logic independent of
  the Streamlit framework.
- Colour palette is kept colourblind-friendly (use tab10 / seaborn defaults).
"""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Use a clean style that works without a display server
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")


# ---------------------------------------------------------------------------
# Moving average
# ---------------------------------------------------------------------------

def add_moving_average(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Add a moving-average column to the DataFrame (sorted by date)."""
    df = df.sort_values("date").copy()
    col_name = f"ma_{window}d"
    df[col_name] = (
        df["consumption_kwh"].rolling(window=window, min_periods=1).mean()
    )
    return df, col_name


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _new_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_daily_consumption(df: pd.DataFrame, ma_window: int = 7) -> plt.Figure:
    """Line chart of daily consumption with an optional moving average."""
    df, ma_col = add_moving_average(df, window=ma_window)

    fig, ax = _new_fig()
    ax.plot(
        df["date"],
        df["consumption_kwh"],
        color="#4C72B0",
        alpha=0.6,
        linewidth=1.2,
        label="Daily consumption",
    )
    if len(df) >= ma_window:
        ax.plot(
            df["date"],
            df[ma_col],
            color="#DD8452",
            linewidth=2,
            label=f"{ma_window}-day moving average",
        )

    # Highlight outliers
    outliers = df[df.get("is_outlier", pd.Series(False, index=df.index))]
    if not outliers.empty:
        ax.scatter(
            outliers["date"],
            outliers["consumption_kwh"],
            color="red",
            zorder=5,
            s=30,
            label="Flagged outlier",
        )

    ax.set_title("Daily Electricity Consumption", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_monthly_bar(monthly_df: pd.DataFrame) -> plt.Figure:
    """Bar chart of monthly consumption totals."""
    fig, ax = _new_fig(figsize=(10, 4))
    x_labels = monthly_df["year_month_str"].tolist()
    values = monthly_df["consumption_kwh"].tolist()

    bars = ax.bar(x_labels, values, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    ax.set_title("Monthly Electricity Consumption (kWh)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Consumption (kWh)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_consumption_histogram(df: pd.DataFrame) -> plt.Figure:
    """Histogram of daily consumption distribution."""
    fig, ax = _new_fig(figsize=(8, 4))
    sns.histplot(
        df["consumption_kwh"],
        bins=20,
        kde=True,
        ax=ax,
        color="#4C72B0",
        edgecolor="white",
    )
    ax.set_title("Distribution of Daily Consumption", fontsize=13, fontweight="bold")
    ax.set_xlabel("Consumption (kWh)")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def plot_rolling_stats(df: pd.DataFrame, window: int = 7) -> plt.Figure:
    """Rolling mean and std chart for trend analysis."""
    df = df.sort_values("date").copy()
    roll = df["consumption_kwh"].rolling(window=window, min_periods=1)
    df["roll_mean"] = roll.mean()
    df["roll_std"] = roll.std().fillna(0)

    fig, ax = _new_fig(figsize=(10, 4))
    ax.plot(df["date"], df["roll_mean"], label=f"{window}-day rolling mean", color="#4C72B0", linewidth=2)
    ax.fill_between(
        df["date"],
        df["roll_mean"] - df["roll_std"],
        df["roll_mean"] + df["roll_std"],
        alpha=0.2,
        color="#4C72B0",
        label=f"±1 std ({window}-day)",
    )
    ax.set_title(f"{window}-Day Rolling Mean ± Std Dev", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_monthly_heatmap(df: pd.DataFrame) -> Optional[plt.Figure]:
    """Calendar heatmap (month × day-of-week) if enough data exists."""
    df = df.copy()
    df["month"] = df["date"].dt.strftime("%b %Y")
    df["dow"] = df["date"].dt.day_name()

    # Need at least 14 days in more than one month to be meaningful
    months = df["date"].dt.to_period("M").unique()
    if len(months) < 2:
        return None

    pivot = df.pivot_table(
        index="month",
        columns="dow",
        values="consumption_kwh",
        aggfunc="mean",
    )
    # Reorder days of week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex(columns=[d for d in day_order if d in pivot.columns])

    fig, ax = _new_fig(figsize=(10, max(3, len(pivot) * 0.5)))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Avg kWh"},
    )
    ax.set_title("Avg Daily Consumption by Month and Day of Week", fontsize=13, fontweight="bold")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Month")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    dates: pd.Series,
    actual: pd.Series,
    predicted: pd.Series,
    title: str = "Actual vs Predicted Consumption",
) -> plt.Figure:
    """Scatter / line comparison of actual and predicted values."""
    fig, ax = _new_fig(figsize=(10, 4))
    ax.plot(dates, actual, label="Actual", color="#4C72B0", linewidth=1.5, marker="o", markersize=3)
    ax.plot(dates, predicted, label="Predicted", color="#DD8452", linewidth=1.5, linestyle="--", marker="x", markersize=4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_future_forecast(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> plt.Figure:
    """Plot historical data with future forecast highlighted."""
    fig, ax = _new_fig(figsize=(12, 4))
    ax.plot(
        history_df["date"],
        history_df["consumption_kwh"],
        label="Historical (actual)",
        color="#4C72B0",
        linewidth=1.5,
    )
    ax.plot(
        forecast_df["date"],
        forecast_df["consumption_kwh"],
        label="Forecast",
        color="#DD8452",
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
    )
    ax.axvline(
        x=history_df["date"].max(),
        color="grey",
        linestyle=":",
        linewidth=1.5,
        label="Forecast start",
    )
    ax.set_title("Historical Consumption and Future Forecast", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Consumption (kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig
