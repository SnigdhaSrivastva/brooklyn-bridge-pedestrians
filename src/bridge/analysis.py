"""Findings: daily/hourly patterns, weather effects controlled for time of day, and an hourly forecast model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def weekday_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Average per hour, not total: weekdays have different numbers of recorded hours,
    so summing (as the original assignment did) ranks days by data coverage, not traffic."""
    out = df.groupby("weekday")["pedestrians"].agg(mean_per_hour="mean", total="sum", hours="count")
    out.index = [DAYS[i] for i in out.index]
    return out.round(1)


def hourly_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Mean pedestrians by weekday x hour of day."""
    table = df.pivot_table(index="weekday", columns="hour", values="pedestrians", aggfunc="mean")
    table.index = [DAYS[i] for i in table.index]
    return table


def weather_effect(df: pd.DataFrame, min_hours: int = 30) -> pd.DataFrame:
    """Traffic under each weather condition relative to clear weather *at the same hour and day type*.

    Raw averages are misleading: rain is more common at night, when the bridge is quiet anyway.
    Comparing within (hour, weekend) cells removes that confounding.
    """
    daytime = df[(df["hour"] >= 7) & (df["hour"] <= 21) & df["weather"].notna()]
    cell = daytime.groupby(["hour", "is_weekend", "weather"])["pedestrians"].agg(["mean", "count"]).reset_index()
    clear = cell[cell["weather"] == "clear"].set_index(["hour", "is_weekend"])["mean"]
    cell = cell.join(clear.rename("clear_mean"), on=["hour", "is_weekend"])
    cell = cell[(cell["count"] >= 3) & cell["clear_mean"].gt(0)]
    cell["ratio"] = cell["mean"] / cell["clear_mean"]
    summary = cell.groupby("weather").apply(
        lambda g: pd.Series({"vs_clear": np.average(g["ratio"], weights=g["count"]), "hours": g["count"].sum()}),
        include_groups=False,
    )
    summary = summary[summary["hours"] >= min_hours]
    summary["change_pct"] = ((summary["vs_clear"] - 1) * 100).round(1)
    return summary.sort_values("vs_clear", ascending=False)


def temperature_curve(df: pd.DataFrame) -> pd.Series:
    """Mean afternoon (12-18h) traffic by 10°F temperature band."""
    afternoon = df[(df["hour"] >= 12) & (df["hour"] <= 18) & df["temperature"].notna()]
    bands = pd.cut(afternoon["temperature"], bins=range(0, 101, 10))
    return afternoon.groupby(bands, observed=True)["pedestrians"].mean().round(0)


FEATURES = ["hour", "weekday", "month", "is_weekend", "is_holiday", "temperature", "precipitation", "has_event"]


@dataclass(frozen=True)
class ModelResult:
    train_rows: int
    test_rows: int
    split_date: str
    baseline_mae: float
    model_mae: float
    model_r2: float

    @property
    def improvement_pct(self) -> float:
        return round((1 - self.model_mae / self.baseline_mae) * 100, 1)


def time_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Chronological split: the model is always evaluated on hours after everything it trained on."""
    ordered = df.sort_values("ts")
    cut = ordered["ts"].iloc[int(len(ordered) * (1 - test_fraction))]
    return ordered[ordered["ts"] < cut], ordered[ordered["ts"] >= cut], cut


def forecast(df: pd.DataFrame, seed: int = 0) -> ModelResult:
    """Gradient-boosted hourly forecast vs a strong seasonal baseline (mean by weekday x hour)."""
    data = df.dropna(subset=["temperature", "precipitation"])
    train, test, cut = time_split(data)

    profile = train.groupby(["weekday", "hour"])["pedestrians"].mean()
    baseline = test.join(profile.rename("pred"), on=["weekday", "hour"])["pred"].fillna(train["pedestrians"].mean())

    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)
    model.fit(train[FEATURES].astype(float), train["pedestrians"])
    pred = np.clip(model.predict(test[FEATURES].astype(float)), 0, None)

    return ModelResult(
        train_rows=len(train), test_rows=len(test), split_date=str(cut.date()),
        baseline_mae=round(mean_absolute_error(test["pedestrians"], baseline), 1),
        model_mae=round(mean_absolute_error(test["pedestrians"], pred), 1),
        model_r2=round(r2_score(test["pedestrians"], pred), 3),
    )
