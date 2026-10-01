"""Load, clean and feature-engineer the NYC DOT Brooklyn Bridge hourly pedestrian counts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

SOURCE_URL = "https://data.cityofnewyork.us/api/views/6fi9-q3ta/rows.csv?accessType=DOWNLOAD"
CACHE = Path(__file__).resolve().parents[2] / "data" / "brooklyn_bridge_pedestrians.csv"

# The raw weather_summary has ~10 labels; group them into conditions with enough data to compare.
WEATHER_GROUPS = {
    "clear-day": "clear", "clear-night": "clear",
    "partly-cloudy-day": "cloudy", "partly-cloudy-night": "cloudy", "cloudy": "cloudy", "fog": "cloudy",
    "wind": "cloudy",
    "rain": "rain", "sleet": "snow", "snow": "snow",
}


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Download once and cache locally so analyses are reproducible offline."""
    if refresh or not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(SOURCE_URL).to_csv(CACHE, index=False)
    return pd.read_csv(CACHE)


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per hour with a valid count.

    The source has duplicate hours and rows missing the count; the original
    assignment kept both, which double-counts some hours.
    """
    df = raw.rename(columns={
        "hour_beginning": "ts", "Pedestrians": "pedestrians",
        "Towards Manhattan": "to_manhattan", "Towards Brooklyn": "to_brooklyn",
    })
    df["ts"] = pd.to_datetime(df["ts"], format="%m/%d/%Y %I:%M:%S %p")
    df = df.dropna(subset=["pedestrians"])
    df = df[df["pedestrians"] >= 0]
    df = df.sort_values("ts").drop_duplicates(subset="ts", keep="last")
    return df[["ts", "pedestrians", "to_manhattan", "to_brooklyn", "weather_summary", "temperature",
               "precipitation", "events"]].reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["ts"].dt.hour
    out["weekday"] = out["ts"].dt.weekday
    out["month"] = out["ts"].dt.month
    out["is_weekend"] = out["weekday"] >= 5
    # Normalize the range start to midnight, or a holiday on the first day would be missed.
    holidays = USFederalHolidayCalendar().holidays(out["ts"].min().normalize(), out["ts"].max())
    out["is_holiday"] = out["ts"].dt.normalize().isin(holidays)
    out["weather"] = out["weather_summary"].map(WEATHER_GROUPS)
    out["has_event"] = out["events"].notna()
    return out


def load(refresh: bool = False) -> pd.DataFrame:
    return add_features(clean(load_raw(refresh)))
