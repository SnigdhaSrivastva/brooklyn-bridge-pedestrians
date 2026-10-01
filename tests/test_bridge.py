from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bridge import analysis
from bridge.data import add_features, clean


def raw_rows(rows: list[tuple[str, float | None, str | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame({
        "hour_beginning": [r[0] for r in rows],
        "location": "Brooklyn Bridge",
        "Pedestrians": [r[1] for r in rows],
        "Towards Manhattan": 0, "Towards Brooklyn": 0,
        "weather_summary": [r[2] for r in rows],
        "temperature": [r[3] for r in rows],
        "precipitation": 0.0, "events": None,
    })


def test_clean_parses_dates_drops_missing_and_dedupes_hours() -> None:
    df = clean(raw_rows([
        ("10/01/2017 03:00:00 PM", 100, "clear-day", 70),
        ("10/01/2017 03:00:00 PM", 120, "clear-day", 70),  # duplicate hour: keep one
        ("10/01/2017 04:00:00 PM", None, "rain", 60),      # missing count: drop
        ("10/01/2017 05:00:00 AM", 5, None, 55),
    ]))
    assert list(df["ts"].dt.hour) == [5, 15]
    assert df["ts"].is_unique
    assert df["pedestrians"].notna().all()


def test_features_weekend_holiday_and_weather_groups() -> None:
    df = add_features(clean(raw_rows([
        ("07/04/2019 02:00:00 PM", 900, "partly-cloudy-day", 85),  # Thursday, Independence Day
        ("07/06/2019 02:00:00 PM", 1500, "rain", 75),              # Saturday
    ])))
    assert list(df["is_holiday"]) == [True, False]
    assert list(df["is_weekend"]) == [False, True]
    assert list(df["weather"]) == ["cloudy", "rain"]


def synthetic(days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Hourly traffic with a known daily shape, weekend boost and a 50% rain penalty."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-01-01", periods=days * 24, freq="h")
    hour = ts.hour.to_numpy()
    base = 200 + 800 * np.exp(-((hour - 15) ** 2) / 18)
    weekend = np.where(ts.weekday >= 5, 1.5, 1.0)
    raining = rng.random(len(ts)) < 0.15
    peds = base * weekend * np.where(raining, 0.5, 1.0) * rng.normal(1, 0.05, len(ts))
    df = pd.DataFrame({
        "ts": ts, "pedestrians": peds.round(), "to_manhattan": 0, "to_brooklyn": 0,
        "weather_summary": np.where(raining, "rain", "clear-day"),
        "temperature": 60 + 10 * np.sin(np.arange(len(ts)) / 500), "precipitation": np.where(raining, 0.1, 0.0),
        "events": None,
    })
    return add_features(df)


def test_weekday_profile_uses_mean_not_total() -> None:
    df = synthetic()
    df = pd.concat([df, df[df["weekday"] == 1]])  # duplicate Tuesdays: totals inflate, means don't
    profile = analysis.weekday_profile(df)
    assert profile.loc["Tue", "total"] > profile.loc["Wed", "total"] * 1.5
    assert profile.loc["Tue", "mean_per_hour"] == pytest.approx(profile.loc["Wed", "mean_per_hour"], rel=0.05)


def test_weather_effect_recovers_the_known_rain_penalty() -> None:
    effect = analysis.weather_effect(synthetic(days=200), min_hours=10)
    assert effect.loc["rain", "change_pct"] == pytest.approx(-50, abs=4)


def test_time_split_never_leaks_future_rows() -> None:
    train, test, cut = analysis.time_split(synthetic())
    assert train["ts"].max() < cut <= test["ts"].min()
    assert len(test) / (len(train) + len(test)) == pytest.approx(0.2, abs=0.01)


def test_forecast_beats_seasonal_baseline_when_weather_matters() -> None:
    result = analysis.forecast(synthetic(days=200))
    assert result.model_mae < result.baseline_mae
    assert result.model_r2 > 0.8
    assert result.improvement_pct > 0


def test_heatmap_shape_and_peak() -> None:
    heat = analysis.hourly_heatmap(synthetic())
    assert heat.shape == (7, 24)
    day, hour = heat.stack().idxmax()
    assert hour in {14, 15, 16}  # the synthetic peak is centred on 15:00, plus noise
    assert day in {"Sat", "Sun"}
