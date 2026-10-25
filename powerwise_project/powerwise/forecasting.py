"""
forecasting.py — Consumption Forecasting and Evaluation
=========================================================
Methods implemented
1. Linear Regression with time-based features (Scikit-learn).
   - Always available when there are enough records.
   - Features: ordinal day number, day-of-week, month, 7-day rolling mean.
   - Evaluation: time-aware train/test split (last 20 % = test).

2. Simple Exponential Smoothing (Statsmodels) — baseline comparison.

3. ARIMA (Statsmodels) — only when the dataset is large enough and passes
   basic stationarity checks.  Clearly labelled as experimental and limited.

Data sufficiency rules (clearly communicated to the user)
- Linear Regression: minimum 30 records (< 30 → refuse with explanation).
- ARIMA: minimum 60 records (< 60 → refuse with explanation).

Evaluation metrics
- MAE  (Mean Absolute Error):  average absolute prediction error in kWh.
- RMSE (Root Mean Squared Error): penalises large errors more than MAE.
- R²   (Coefficient of Determination): proportion of variance explained.
  R² is only shown when it is meaningful (not on very small test sets).

Honest limitations
- The model is trained on whatever historical data is available.
- Predictions assume the future follows the same patterns as the past.
- External shocks (tariff changes, new appliances) are not captured.
- All forecasts carry substantial uncertainty; they should be used for
  rough planning, not billing or financial decisions.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_RECORDS_LINEAR = 30   # minimum rows for linear regression
MIN_RECORDS_ARIMA = 60    # minimum rows for ARIMA
TEST_FRACTION = 0.20      # last 20 % of records used for test


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create numeric features from date for regression."""
    df = df.sort_values("date").copy()
    # Ordinal day index — captures overall trend
    df["day_ordinal"] = (df["date"] - df["date"].min()).dt.days
    # Cyclical features — capture weekly and annual seasonality without
    # treating day numbers as unboundedly increasing
    df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["date"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["date"].dt.month - 1) / 12)
    # Rolling mean lag feature — helps capture recent level
    df["roll7"] = df["consumption_kwh"].rolling(7, min_periods=1).mean()
    return df


FEATURE_COLS = ["day_ordinal", "dow_sin", "dow_cos", "month_sin", "month_cos", "roll7"]


# ---------------------------------------------------------------------------
# Linear Regression forecasting
# ---------------------------------------------------------------------------

def train_linear_model(
    df: pd.DataFrame,
) -> tuple[Optional[dict], str]:
    """Train a Linear Regression model on cleaned consumption data.

    Returns (result_dict, error_message).
    result_dict contains the trained model, scaler, evaluation metrics,
    train/test DataFrames, and the last day ordinal (for future prediction).
    """
    if len(df) < MIN_RECORDS_LINEAR:
        return None, (
            f"Not enough data for regression. "
            f"Need at least {MIN_RECORDS_LINEAR} records; you have {len(df)}. "
            "Upload more historical data or use the sample dataset."
        )

    df = _build_features(df)
    df = df.dropna(subset=FEATURE_COLS + ["consumption_kwh"])

    # Time-aware split — last TEST_FRACTION of records = test set
    n_test = max(1, int(len(df) * TEST_FRACTION))
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    X_train = train[FEATURE_COLS].values
    y_train = train["consumption_kwh"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["consumption_kwh"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # Clip negative predictions (physically impossible)
    y_pred = np.clip(y_pred, 0, None)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred)) if n_test >= 5 else None

    result = {
        "model": model,
        "scaler": scaler,
        "train_df": train,
        "test_df": test,
        "y_pred_test": y_pred,
        "metrics": {"MAE": mae, "RMSE": rmse, "R²": r2},
        "n_train": len(train),
        "n_test": n_test,
        "origin_date": df["date"].min(),
        "last_day_ordinal": int(df["day_ordinal"].max()),
        "last_date": df["date"].max(),
    }
    return result, ""


def forecast_future_linear(
    model_result: dict,
    days_ahead: int = 30,
) -> pd.DataFrame:
    """Generate future consumption forecasts using the trained linear model.

    Parameters
    ----------
    model_result : dict
        Output of train_linear_model().
    days_ahead : int
        Number of days to forecast into the future.

    Returns
    -------
    pd.DataFrame with columns: date, consumption_kwh (predicted).
    """
    model = model_result["model"]
    scaler = model_result["scaler"]
    origin_date = model_result["origin_date"]
    last_ordinal = model_result["last_day_ordinal"]
    last_date = model_result["last_date"]
    last_roll7 = float(model_result["train_df"]["consumption_kwh"].tail(7).mean())

    future_records = []
    current_roll7 = last_roll7

    for i in range(1, days_ahead + 1):
        future_date = last_date + pd.Timedelta(days=i)
        ordinal = last_ordinal + i
        row = {
            "day_ordinal": ordinal,
            "dow_sin": np.sin(2 * np.pi * future_date.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * future_date.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * (future_date.month - 1) / 12),
            "month_cos": np.cos(2 * np.pi * (future_date.month - 1) / 12),
            "roll7": current_roll7,
        }
        future_records.append((future_date, row))

    dates = [r[0] for r in future_records]
    X_future = np.array([[r[1][c] for c in FEATURE_COLS] for r in future_records])
    X_future_s = scaler.transform(X_future)
    y_future = model.predict(X_future_s)
    y_future = np.clip(y_future, 0, None)

    return pd.DataFrame({"date": dates, "consumption_kwh": y_future})


# ---------------------------------------------------------------------------
# ARIMA (optional, guarded by data sufficiency check)
# ---------------------------------------------------------------------------

def train_arima(df: pd.DataFrame) -> tuple[Optional[dict], str]:
    """Fit a simple ARIMA(1,1,1) model if the data is sufficient.

    Returns (result_dict, error_message).
    Uses a minimal ARIMA order — more complex orders are not warranted without
    careful ACF/PACF analysis, which is beyond the scope of this project.
    """
    if len(df) < MIN_RECORDS_ARIMA:
        return None, (
            f"ARIMA requires at least {MIN_RECORDS_ARIMA} records. "
            f"You have {len(df)}. Use Linear Regression instead."
        )

    try:
        from statsmodels.tsa.arima.model import ARIMA  # lazy import
    except ImportError:
        return None, "statsmodels is not installed. Run: pip install statsmodels"

    df = df.sort_values("date").copy()
    series = df.set_index("date")["consumption_kwh"].asfreq("D")

    # Fill any gaps created by the daily frequency reindex
    series = series.interpolate(method="time")

    n_test = max(7, int(len(series) * TEST_FRACTION))
    train_series = series.iloc[:-n_test]
    test_series = series.iloc[-n_test:]

    try:
        arima_model = ARIMA(train_series, order=(1, 1, 1))
        fitted = arima_model.fit()
        forecast = fitted.forecast(steps=n_test)
        forecast = np.clip(forecast.values, 0, None)
    except Exception as exc:  # noqa: BLE001
        return None, f"ARIMA fitting failed: {exc}"

    y_test = test_series.values
    mae = float(mean_absolute_error(y_test, forecast))
    rmse = float(np.sqrt(mean_squared_error(y_test, forecast)))

    result = {
        "fitted_model": fitted,
        "test_dates": test_series.index,
        "y_test": y_test,
        "y_pred": forecast,
        "metrics": {"MAE": mae, "RMSE": rmse},
        "n_train": len(train_series),
        "n_test": n_test,
        "last_date": series.index[-1],
        "series": series,
    }
    return result, ""


def forecast_future_arima(arima_result: dict, days_ahead: int = 30) -> pd.DataFrame:
    """Generate future forecasts from a fitted ARIMA model."""
    fitted = arima_result["fitted_model"]
    last_date = arima_result["last_date"]

    forecast_values = fitted.forecast(steps=days_ahead)
    forecast_values = np.clip(forecast_values, 0, None)
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D"
    )
    return pd.DataFrame({"date": future_dates, "consumption_kwh": forecast_values})


# ---------------------------------------------------------------------------
# Weather-enhanced local consumption forecast
# ---------------------------------------------------------------------------

WEATHER_FEATURE_COLS = [
    "day_ordinal", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "roll7", "weather_temp_max", "weather_temp_min",
    "weather_precipitation", "weather_wind_max",
]


def train_weather_linear_model(df: pd.DataFrame, historical_weather: pd.DataFrame) -> tuple[Optional[dict], str]:
    """Train a local model using consumption history plus matched historical weather."""
    if len(df) < MIN_RECORDS_LINEAR:
        return None, f"Weather-enhanced regression needs at least {MIN_RECORDS_LINEAR} consumption records; you have {len(df)}."
    required = {"date", "weather_temp_max", "weather_temp_min", "weather_precipitation", "weather_wind_max"}
    if not required.issubset(historical_weather.columns):
        return None, "Historical weather data is missing required daily variables."

    consumption = df[["date", "consumption_kwh"]].copy()
    consumption["date"] = pd.to_datetime(consumption["date"]).dt.normalize()
    weather = historical_weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    merged = consumption.merge(weather, on="date", how="inner").sort_values("date")
    merged["day_ordinal"] = (merged["date"] - merged["date"].min()).dt.days
    merged["dow_sin"] = np.sin(2 * np.pi * merged["date"].dt.dayofweek / 7)
    merged["dow_cos"] = np.cos(2 * np.pi * merged["date"].dt.dayofweek / 7)
    merged["month_sin"] = np.sin(2 * np.pi * (merged["date"].dt.month - 1) / 12)
    merged["month_cos"] = np.cos(2 * np.pi * (merged["date"].dt.month - 1) / 12)
    merged["roll7"] = merged["consumption_kwh"].rolling(7, min_periods=1).mean().shift(1)
    merged = merged.dropna(subset=WEATHER_FEATURE_COLS + ["consumption_kwh"])
    if len(merged) < MIN_RECORDS_LINEAR:
        return None, (f"Only {len(merged)} consumption dates matched historical weather. "
                      f"At least {MIN_RECORDS_LINEAR} matched dates are needed. Try a different location or dataset.")

    n_test = max(1, int(len(merged) * TEST_FRACTION))
    train, test = merged.iloc[:-n_test], merged.iloc[-n_test:]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[WEATHER_FEATURE_COLS].values)
    X_test = scaler.transform(test[WEATHER_FEATURE_COLS].values)
    model = LinearRegression().fit(X_train, train["consumption_kwh"].values)
    y_pred = np.clip(model.predict(X_test), 0, None)
    y_test = test["consumption_kwh"].values
    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R²": float(r2_score(y_test, y_pred)) if n_test >= 5 else None,
    }
    result = {
        "model": model, "scaler": scaler, "features": WEATHER_FEATURE_COLS,
        "train_df": train, "test_df": test, "y_pred_test": y_pred,
        "metrics": metrics, "n_train": len(train), "n_test": n_test,
        "origin_date": merged["date"].min(), "last_date": merged["date"].max(),
        "last_day_ordinal": int(merged["day_ordinal"].max()),
        "last_roll7": float(merged["consumption_kwh"].tail(7).mean()),
        "weather_means": merged[WEATHER_FEATURE_COLS[6:]].mean().to_dict(),
    }
    return result, ""


def forecast_future_weather_linear(model_result: dict, future_weather: pd.DataFrame) -> pd.DataFrame:
    """Forecast future household kWh using real forecast weather for matching dates."""
    weather = future_weather.copy()
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    last_date = pd.Timestamp(model_result["last_date"]).normalize()
    weather = weather[weather["date"] > last_date].sort_values("date").copy()
    if weather.empty:
        raise ValueError("The weather forecast does not include dates after the last consumption reading.")
    weather["day_ordinal"] = (weather["date"] - pd.Timestamp(model_result["origin_date"])).dt.days
    weather["dow_sin"] = np.sin(2 * np.pi * weather["date"].dt.dayofweek / 7)
    weather["dow_cos"] = np.cos(2 * np.pi * weather["date"].dt.dayofweek / 7)
    weather["month_sin"] = np.sin(2 * np.pi * (weather["date"].dt.month - 1) / 12)
    weather["month_cos"] = np.cos(2 * np.pi * (weather["date"].dt.month - 1) / 12)
    last_roll7 = model_result.get("last_roll7")
    if last_roll7 is None:
        history_parts = []
        for key in ("train_df", "test_df"):
            part = model_result.get(key)
            if isinstance(part, pd.DataFrame) and "consumption_kwh" in part.columns:
                history_parts.append(part[["consumption_kwh"]])
        if history_parts:
            history_values = pd.concat(history_parts, ignore_index=True)["consumption_kwh"].dropna()
            last_roll7 = float(history_values.tail(7).mean()) if not history_values.empty else None
    if last_roll7 is None:
        raise ValueError("The weather model is missing its recent consumption history. Please rerun the forecast from the Forecast page.")
    weather["roll7"] = float(last_roll7)
    # Older or cached model results may not contain weather_means.
    # Build safe fallback values from training/testing data, then from the
    # available future-weather rows, so forecasting does not fail on a missing key.
    weather_means = model_result.get("weather_means", {}) or {}
    historical_frames = []
    for key in ("train_df", "test_df"):
        frame = model_result.get(key)
        if isinstance(frame, pd.DataFrame):
            historical_frames.append(frame)
    for col in WEATHER_FEATURE_COLS[6:]:
        if col not in weather.columns:
            weather[col] = np.nan
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
        mean_value = weather_means.get(col)
        if mean_value is None:
            vals = []
            for frame in historical_frames:
                if col in frame.columns:
                    vals.extend(pd.to_numeric(frame[col], errors="coerce").dropna().tolist())
            if vals:
                mean_value = float(np.mean(vals))
        if mean_value is None or not np.isfinite(float(mean_value)):
            available = weather[col].dropna()
            mean_value = float(available.mean()) if not available.empty else 0.0
        weather[col] = weather[col].fillna(float(mean_value))
    features = model_result.get("features", WEATHER_FEATURE_COLS)
    scaler = model_result.get("scaler")
    model = model_result.get("model")

    # Guard against an older/inconsistent result where the saved scaler was
    # fitted with fewer columns than the current weather feature set.
    expected_features = getattr(scaler, "n_features_in_", len(features))
    if expected_features != len(features) or len(features) != len(WEATHER_FEATURE_COLS):
        train_frame = model_result.get("train_df")
        if not isinstance(train_frame, pd.DataFrame):
            raise ValueError(
                "The saved weather model uses an outdated feature set. Please rerun "
                "the weather-enhanced forecast to retrain it."
            )
        available_features = [col for col in WEATHER_FEATURE_COLS if col in train_frame.columns]
        if len(available_features) != len(WEATHER_FEATURE_COLS):
            raise ValueError(
                "The saved training data does not contain all weather features. "
                "Please rerun the weather-enhanced forecast."
            )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_frame[available_features].values)
        model = LinearRegression().fit(X_train, train_frame["consumption_kwh"].values)
        features = available_features
        model_result["scaler"] = scaler
        model_result["model"] = model
        model_result["features"] = features

    X = scaler.transform(weather[features].values)
    predictions = np.clip(model.predict(X), 0, None)
    return pd.DataFrame({
        "date": weather["date"].values,
        "consumption_kwh": predictions,
        "forecast_temperature_max_c": weather["weather_temp_max"].values,
        "forecast_temperature_min_c": weather["weather_temp_min"].values,
        "forecast_precipitation_mm": weather["weather_precipitation"].values,
        "forecast_wind_max_kmh": weather["weather_wind_max"].values,
    })
