"""Run every analysis, save figures to figures/, and print the findings used in the README."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bridge import analysis
from bridge.data import load

FIGURES = Path(__file__).resolve().parents[2] / "figures"


def main() -> None:
    df = load()
    FIGURES.mkdir(exist_ok=True)
    print(f"{len(df):,} clean hourly rows, {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}\n")

    profile = analysis.weekday_profile(df)
    print("Average pedestrians per hour by weekday (vs. raw totals):\n", profile, "\n")

    heat = analysis.hourly_heatmap(df)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(0, 24, 2), [f"{h}:00" for h in range(0, 24, 2)])
    ax.set_yticks(range(7), heat.index)
    ax.set_title("Brooklyn Bridge: average pedestrians per hour")
    fig.colorbar(im, ax=ax, label="pedestrians / hour")
    fig.tight_layout()
    fig.savefig(FIGURES / "weekday_hour_heatmap.png", dpi=150)
    plt.close(fig)
    peak_day, peak_hour = heat.stack().idxmax()
    print(f"Busiest slot: {peak_day} {peak_hour}:00 ({heat.stack().max():.0f}/hour)\n")

    weather = analysis.weather_effect(df)
    print("Daytime traffic vs. clear weather, same hour & day type:\n", weather, "\n")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = ["#2b8a3e" if v >= 0 else "#c92a2a" for v in weather["change_pct"]]
    ax.barh(weather.index, weather["change_pct"], color=colors)
    ax.axvline(0, color="#555", lw=0.8)
    ax.set_xlabel("% change vs clear weather (controlled for hour & weekend)")
    ax.set_title("Weather effect on foot traffic")
    fig.tight_layout()
    fig.savefig(FIGURES / "weather_effect.png", dpi=150)
    plt.close(fig)

    temp = analysis.temperature_curve(df)
    print("Afternoon traffic by temperature (°F):\n", temp, "\n")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot([i.mid for i in temp.index], temp.values, marker="o")
    ax.set_xlabel("temperature (°F)")
    ax.set_ylabel("pedestrians / hour (12-18h)")
    ax.set_title("Warmer afternoons draw more walkers")
    fig.tight_layout()
    fig.savefig(FIGURES / "temperature_curve.png", dpi=150)
    plt.close(fig)

    result = analysis.forecast(df)
    print(f"Hourly forecast, trained before {result.split_date} ({result.train_rows:,} rows), "
          f"tested after ({result.test_rows:,} rows):")
    print(f"  seasonal baseline MAE {result.baseline_mae}  |  gradient boosting MAE {result.model_mae} "
          f"(R² {result.model_r2})  ->  {result.improvement_pct}% lower error")


if __name__ == "__main__":
    main()
