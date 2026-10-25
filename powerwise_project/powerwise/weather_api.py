"""Free weather data integration using Open-Meteo (no API key required)."""
from __future__ import annotations

from typing import Any

import pandas as pd
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 15


class WeatherAPIError(RuntimeError):
    """Raised when weather data cannot be fetched or parsed."""


def geocode_location(location: str) -> dict[str, Any]:
    """Resolve a place name to the first matching location."""
    if not location or not location.strip():
        raise WeatherAPIError("Enter a city or location name.")
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": location.strip(), "count": 5, "language": "en", "format": "json"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Could not reach Open-Meteo geocoding: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError("The geocoding service returned invalid JSON.") from exc

    results = payload.get("results") or []
    if not results:
        raise WeatherAPIError(f"No location found for '{location}'. Try a nearby city name.")
    return results[0]


def fetch_weather(latitude: float, longitude: float, days: int = 7) -> dict[str, Any]:
    """Fetch current conditions and a daily forecast for coordinates."""
    days = max(1, min(int(days), 7))
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
        "forecast_days": days,
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Could not fetch weather data: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError("The weather service returned invalid JSON.") from exc

    if not isinstance(payload, dict) or "current" not in payload or "daily" not in payload:
        raise WeatherAPIError("The weather service response did not contain expected current and daily data.")
    return payload


def weather_code_description(code: int | float | None) -> str:
    """Convert WMO weather codes to short human-readable descriptions."""
    if code is None:
        return "Unknown"
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Unknown"
    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return {1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast"}[code]
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55):
        return "Drizzle"
    if code in (56, 57):
        return "Freezing drizzle"
    if code in (61, 63, 65):
        return "Rain"
    if code in (66, 67):
        return "Freezing rain"
    if code in (71, 73, 75, 77):
        return "Snow"
    if code in (80, 81, 82):
        return "Rain showers"
    if code in (85, 86):
        return "Snow showers"
    if code == 95:
        return "Thunderstorm"
    if code in (96, 99):
        return "Thunderstorm with hail"
    return "Unknown conditions"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_historical_weather(latitude: float, longitude: float, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily historical weather aligned to household meter-reading dates."""
    import pandas as pd

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "timezone": "auto",
    }
    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise WeatherAPIError(f"Could not fetch historical weather: {exc}") from exc
    except ValueError as exc:
        raise WeatherAPIError("The historical weather service returned invalid JSON.") from exc

    daily = payload.get("daily") if isinstance(payload, dict) else None
    if not daily or not daily.get("time"):
        raise WeatherAPIError("No historical weather data was returned for this date range.")
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).normalize(),
        "weather_temp_max": daily.get("temperature_2m_max", []),
        "weather_temp_min": daily.get("temperature_2m_min", []),
        "weather_precipitation": daily.get("precipitation_sum", []),
        "weather_wind_max": daily.get("wind_speed_10m_max", []),
    })


def daily_forecast_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert Open-Meteo daily forecast response to model feature columns."""
    daily = payload.get("daily", {})
    if not daily or not daily.get("time"):
        raise WeatherAPIError("No daily weather forecast is available.")
    import pandas as pd
    return pd.DataFrame({
        "date": pd.to_datetime(daily["time"]).normalize(),
        "weather_temp_max": daily.get("temperature_2m_max", []),
        "weather_temp_min": daily.get("temperature_2m_min", []),
        "weather_precipitation": daily.get("precipitation_sum", []),
        "weather_wind_max": daily.get("wind_speed_10m_max", []),
    })
