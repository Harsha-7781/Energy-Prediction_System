"""
Step 6 - Seasonal trend analysis.

Decomposes the cleaned hourly energy series into trend / seasonal /
residual components using STL, at two resolutions that match the two
seasonal cycles EDA (step 4) found:

  1. Hourly, period=24 -> the dominant daily occupancy cycle (per floor).
  2. Daily-aggregated, period=7 -> the weekly weekday/weekend cycle
     (combined building total), whose STL *trend* component also
     reveals the slower month-to-month movement already glimpsed in
     reports/figures/05_monthly_seasonal.png.

Uses data/processed/cleaned_hourly.csv (post step-5 cleaning) so the
decomposition isn't distorted by the sensor-fault spikes step 5 removed.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import STL

IN_PATH = Path("data/processed/cleaned_hourly.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/seasonal_decomposition.md")


def load() -> pd.DataFrame:
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    return df


def floor_hourly_series(df: pd.DataFrame, floor: int) -> pd.Series:
    sub = df[df.floor_id == floor].set_index("timestamp")["energy"].sort_index()
    sub = sub.asfreq("h")  # each floor's own range is already contiguous; this just makes it explicit
    return sub


def plot_stl(result, title: str, out_path: Path):
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    result.observed.plot(ax=axes[0], linewidth=0.7)
    axes[0].set_ylabel("Observed")
    result.trend.plot(ax=axes[1], linewidth=1, color="tab:orange")
    axes[1].set_ylabel("Trend")
    result.seasonal.plot(ax=axes[2], linewidth=0.5, color="tab:green")
    axes[2].set_ylabel("Seasonal")
    result.resid.plot(ax=axes[3], linewidth=0.5, color="tab:red")
    axes[3].set_ylabel("Residual")
    axes[3].set_xlabel("Timestamp")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    df = load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Hourly STL (daily cycle), per floor ---
    hourly_stats = []
    trend_by_floor = {}
    example_result = None
    for floor in sorted(df.floor_id.unique()):
        series = floor_hourly_series(df, floor)
        stl = STL(series, period=24, robust=True)
        result = stl.fit()
        trend_by_floor[floor] = result.trend

        var_obs = series.var()
        var_seasonal = result.seasonal.var()
        var_resid = result.resid.var()
        hourly_stats.append({
            "floor_id": floor,
            "seasonal_share": round(var_seasonal / var_obs, 3),
            "residual_share": round(var_resid / var_obs, 3),
            "trend_min": round(result.trend.min(), 1),
            "trend_max": round(result.trend.max(), 1),
        })
        if floor == 3:  # representative floor with known outlier/imputation activity
            example_result = result
            plot_stl(result, f"STL decomposition (daily cycle, period=24h) - Floor {floor}",
                     FIG_DIR / "07_stl_hourly_floor3.png")

    # Overlay of trend components across floors (slower seasonal movement)
    fig, ax = plt.subplots(figsize=(13, 5))
    for floor, trend in trend_by_floor.items():
        ax.plot(trend.index, trend.values, label=f"Floor {floor}", linewidth=0.8)
    ax.set_title("STL trend component (daily-cycle decomposition), all floors")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Trend (deseasonalized level)")
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_stl_trend_all_floors.png", dpi=120)
    plt.close(fig)

    # --- 2. Daily-aggregated STL (weekly cycle), combined building total ---
    daily_total = df.groupby(df["timestamp"].dt.date)["energy"].sum()
    daily_total.index = pd.to_datetime(daily_total.index)
    daily_total = daily_total.asfreq("D")
    stl_daily = STL(daily_total, period=7, robust=True)
    result_daily = stl_daily.fit()
    plot_stl(result_daily, "STL decomposition (weekly cycle, period=7d) - combined building total",
             FIG_DIR / "09_stl_daily_combined.png")

    var_obs_d = daily_total.var()
    weekly_seasonal_share = round(result_daily.seasonal.var() / var_obs_d, 3)
    weekly_residual_share = round(result_daily.resid.var() / var_obs_d, 3)
    trend_peak_date = result_daily.trend.idxmax().date()
    trend_trough_date = result_daily.trend.idxmin().date()

    # --- Report ---
    hourly_df = pd.DataFrame(hourly_stats).set_index("floor_id")
    lines = [
        "# Seasonal Trend Decomposition",
        "",
        "STL decomposition of `data/processed/cleaned_hourly.csv` "
        "(post-cleaning). Two resolutions, matching the two cycles EDA "
        "found: hourly/period=24 (daily occupancy cycle, per floor) and "
        "daily/period=7 (weekly cycle, combined building total).",
        "",
        "## Daily cycle (hourly STL, period=24h)",
        "",
        "`seasonal_share` / `residual_share` = fraction of that floor's "
        "variance explained by the recurring daily pattern vs. leftover "
        "noise -- higher seasonal_share means the floor's load is more "
        "predictably occupancy-driven.",
        "",
        hourly_df.to_markdown(),
        "",
        "Figures: `07_stl_hourly_floor3.png` (worked example), "
        "`08_stl_trend_all_floors.png` (deseasonalized trend level, all "
        "floors overlaid -- this is the cleanest view of each floor's "
        "*underlying* load level with the daily cycle removed).",
        "",
        "## Weekly cycle (daily STL, period=7d, combined total)",
        "",
        f"- Seasonal (weekly) share of variance: {weekly_seasonal_share}",
        f"- Residual share of variance: {weekly_residual_share}",
        f"- Trend component peaks around **{trend_peak_date}** and troughs "
        f"around **{trend_trough_date}**.",
        "- This trend component is the deseasonalized (day-of-week "
        "removed) view of the monthly pattern already visible in "
        "`05_monthly_seasonal.png` -- consistent with a cool-season dip "
        "and hot-season rise driven by AC load, now isolated from the "
        "weekday/weekend swing.",
        "- Two dips stand out beyond the ordinary seasonal wave: a sharp "
        "drop to ~3,900 around **late Jan-early Feb 2019**, plausibly a "
        "university low-occupancy/exam-break period rather than a data "
        "artifact (all 7 floors are present and contributing in this "
        "window). Worth flagging as a hypothesis, not a confirmed cause "
        "-- we have no occupancy calendar to verify it against.",
        "- **Important caveat**: the trend also steps down and flattens "
        "from late October 2019 onward. This coincides exactly with "
        "Floor 6's data truncation (2019-10-22, see "
        "`data_inspection.md`) -- from that date the 'combined total' is "
        "a sum over only 6 floors instead of 7, which mechanically lowers "
        "it regardless of any real occupancy change. **This is a "
        "floor-count artifact, not a genuine seasonal decline** -- don't "
        "read the Nov-Dec 2019 section of this trend as 'the building "
        "used less energy'; it means 'one fewer floor is represented'.",
        "",
        "Figure: `09_stl_daily_combined.png`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(hourly_df)
    print(f"\nWeekly seasonal share: {weekly_seasonal_share}, residual share: {weekly_residual_share}")
    print(f"Daily-cycle trend peak/trough dates: {trend_peak_date} / {trend_trough_date}")
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
