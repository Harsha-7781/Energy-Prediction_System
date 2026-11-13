"""
Exploratory data analysis of building energy consumption patterns.

Analyzes hourly consumption patterns (daily / weekly / seasonal) from
data/processed/combined_hourly.csv, per floor and combined, and saves
plots to reports/figures/. This is exploratory only -- no cleaning or
imputation happens here (that's step 5); NaNs are simply skipped by the
aggregation functions used (pandas mean/sum default skipna=True).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

IN_PATH = Path("data/processed/combined_hourly.csv")
FIG_DIR = Path("reports/figures")

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.to_period("M")
    df["is_weekend"] = df["dow"] >= 5
    return df


def fig_daily_total(df: pd.DataFrame):
    daily = df.groupby("date")["energy"].sum()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(daily.index), daily.values, linewidth=0.8)
    ax.set_title("Combined building: total daily energy (all floors summed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily energy (sum of hourly kW readings)")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_daily_total_energy.png", dpi=120)
    plt.close(fig)


def fig_per_floor_daily(df: pd.DataFrame):
    daily = df.groupby(["date", "floor_id"])["energy"].mean().unstack("floor_id")
    fig, ax = plt.subplots(figsize=(14, 6))
    for floor in sorted(daily.columns):
        ax.plot(pd.to_datetime(daily.index), daily[floor], label=f"Floor {floor}", linewidth=0.8)
    ax.set_title("Per-floor daily mean hourly energy over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean hourly energy that day")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_per_floor_daily_energy.png", dpi=120)
    plt.close(fig)


def fig_hourly_profile(df: pd.DataFrame):
    profile = df.groupby(["floor_id", "is_weekend", "hour"])["energy"].mean().reset_index()
    fig, axes = plt.subplots(1, 7, figsize=(20, 4), sharey=False)
    for ax, floor in zip(axes, sorted(df["floor_id"].unique())):
        sub = profile[profile.floor_id == floor]
        for wk, label, style in [(False, "Weekday", "-"), (True, "Weekend", "--")]:
            s = sub[sub.is_weekend == wk].sort_values("hour")
            ax.plot(s["hour"], s["energy"], style, label=label)
        ax.set_title(f"Floor {floor}")
        ax.set_xlabel("Hour")
    axes[0].set_ylabel("Mean energy")
    axes[0].legend()
    fig.suptitle("Average hourly consumption profile: weekday vs weekend, per floor")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_hourly_profile_weekday_weekend.png", dpi=120)
    plt.close(fig)


def fig_day_of_week(df: pd.DataFrame):
    dow = df.groupby(["floor_id", "dow"])["energy"].mean().unstack("floor_id")
    fig, ax = plt.subplots(figsize=(10, 5))
    dow.plot(kind="bar", ax=ax)
    ax.set_xticklabels(DOW_LABELS, rotation=0)
    ax.set_title("Average energy by day of week, per floor")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Mean hourly energy")
    ax.legend(title="Floor", ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_day_of_week.png", dpi=120)
    plt.close(fig)


def fig_monthly_seasonal(df: pd.DataFrame):
    monthly = df.groupby(["month", "floor_id"])["energy"].mean().unstack("floor_id")
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly.plot(ax=ax, marker="o", markersize=3)
    ax.set_title("Monthly mean hourly energy, per floor (seasonal view, Jul 2018 - Dec 2019)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean hourly energy")
    ax.legend(title="Floor", ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_monthly_seasonal.png", dpi=120)
    plt.close(fig)


def fig_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [df[df.floor_id == f]["energy"].dropna().values for f in sorted(df.floor_id.unique())]
    ax.boxplot(data, tick_labels=[f"F{f}" for f in sorted(df.floor_id.unique())], showfliers=False)
    ax.set_title("Distribution of hourly energy per floor (outliers hidden for scale)")
    ax.set_ylabel("Hourly energy")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_energy_distribution_per_floor.png", dpi=120)
    plt.close(fig)


def print_summary(df: pd.DataFrame):
    print("Combined hourly data summary")
    print(f"  Rows: {len(df):,}, Range: {df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"  NaN energy rows: {df['energy'].isna().sum():,} ({df['energy'].isna().mean()*100:.1f}%)")
    print("\nMean hourly energy per floor:")
    print(df.groupby("floor_id")["energy"].mean().round(1))
    print("\nMean hourly energy: weekday vs weekend (combined):")
    print(df.groupby("is_weekend")["energy"].mean().round(1))
    peak_hour = df.groupby("hour")["energy"].mean().idxmax()
    trough_hour = df.groupby("hour")["energy"].mean().idxmin()
    print(f"\nCombined peak hour: {peak_hour}:00, trough hour: {trough_hour}:00")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print_summary(df)
    fig_daily_total(df)
    fig_per_floor_daily(df)
    fig_hourly_profile(df)
    fig_day_of_week(df)
    fig_monthly_seasonal(df)
    fig_distribution(df)
    print(f"\nSaved 6 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
