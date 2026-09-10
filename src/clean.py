"""
Step 5 - Data cleaning.

Handles missing values and abnormal readings in data/processed/combined_hourly.csv,
producing data/processed/cleaned_hourly.csv. Logic is justified by what
step 2 (reports/data_inspection.md) and step 4 (reports/eda_findings.md)
found, not applied blindly:

- Step 2 found NO negative kW readings and a complete minute-level
  timestamp index everywhere, but several AC columns with sensor-fault-
  style extreme spikes (stuck-high readings), and highly variable NaN
  rates concentrated in specific floor/column/period combinations rather
  than uniform noise.
- Step 4 confirmed hourly energy has strong, regular daily/weekly
  structure (occupancy-driven), which we exploit for imputation: a
  missing Tuesday-9am reading is best estimated from other Tuesday-9am
  readings on that floor, not from a global mean.

Cleaning logic:
1. Low-coverage hours (pct_minutes_available < 50%) are treated as
   unreliable and blanked before imputation -- an "hourly mean" built
   from only a couple of real minutes is not a trustworthy label.
2. Per-floor outlier detection on hourly energy using median +/- 3xIQR
   (a wide, "extreme outlier" multiplier, not the standard 1.5x mild
   threshold) -- chosen deliberately wide so genuine peak-demand hours
   survive and only implausible spikes (consistent with the sensor-fault
   spikes step 2 documented) get flagged.
3. Both blanked-low-coverage and flagged-outlier energy values are set
   to NaN and imputed the same way as genuinely missing hours:
   a. time-interpolate gaps of up to 3 consecutive hours per floor
      (short gaps -- safe to interpolate linearly).
   b. remaining gaps filled with that floor's median for the same
      (hour-of-day, day-of-week) -- a seasonal-naive fill that respects
      the strong occupancy pattern EDA found, rather than a flat mean.
4. Sensor columns (temperature/humidity/lux) get a light short-gap
   interpolation only (limit=6h); Floor 1 has no sensors at all (100%
   missing, by construction -- see step 3) and is left NaN here, to be
   handled explicitly at feature-engineering time rather than faked here.

Every row keeps `energy_raw` (pre-cleaning value, NaN where genuinely
missing) and boolean `was_outlier` / `was_imputed` flags so the model and
any report can distinguish observed from filled values.
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/combined_hourly.csv")
OUT_PATH = Path("data/processed/cleaned_hourly.csv")
REPORT_PATH = Path("reports/cleaning_report.md")

LOW_COVERAGE_PCT = 50.0
IQR_MULTIPLIER = 3.0
INTERP_LIMIT_HOURS = 3
SENSOR_INTERP_LIMIT_HOURS = 6


def flag_outliers(group: pd.DataFrame) -> pd.Series:
    energy = group["energy"]
    q1, q3 = energy.quantile(0.25), energy.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - IQR_MULTIPLIER * iqr
    upper = q3 + IQR_MULTIPLIER * iqr
    return (energy < lower) | (energy > upper)


def seasonal_fill(group: pd.DataFrame, col: str) -> pd.Series:
    hour = group["timestamp"].dt.hour
    dow = group["timestamp"].dt.dayofweek
    medians = group.groupby([hour, dow])[col].transform("median")
    filled = group[col].fillna(medians)
    return filled


def clean_floor(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").reset_index(drop=True)
    group["energy_raw"] = group["energy"]

    low_coverage = group["pct_minutes_available"] < LOW_COVERAGE_PCT
    is_outlier = flag_outliers(group) & group["energy"].notna()

    group["was_outlier"] = is_outlier
    group["was_low_coverage"] = low_coverage

    working = group["energy"].copy()
    working[low_coverage | is_outlier] = np.nan

    was_missing_before_interp = working.isna()
    interpolated = working.interpolate(
        method="linear", limit=INTERP_LIMIT_HOURS, limit_area="inside"
    )
    group["energy"] = interpolated
    group["energy"] = seasonal_fill(group, "energy")

    # Last-resort fallback (only hit if an (hour, dow) bucket is 100% NaN
    # for this floor, e.g. a bucket that only ever existed during the
    # missing window) -- floor-level overall median.
    remaining = group["energy"].isna()
    if remaining.any():
        group.loc[remaining, "energy"] = group["energy"].median()

    group["was_imputed"] = was_missing_before_interp | remaining

    for sensor_col in ("temperature", "humidity", "lux"):
        if sensor_col in group.columns and group[sensor_col].notna().any():
            group[sensor_col] = group[sensor_col].interpolate(
                method="linear", limit=SENSOR_INTERP_LIMIT_HOURS, limit_area="inside"
            )

    return group


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    n_before_missing = df["energy"].isna().sum()

    parts = [clean_floor(df[df["floor_id"] == floor].copy()) for floor in sorted(df["floor_id"].unique())]
    cleaned = pd.concat(parts, ignore_index=True)
    cleaned = cleaned.sort_values(["floor_id", "timestamp"]).reset_index(drop=True)

    n_outliers = int(cleaned["was_outlier"].sum())
    n_low_coverage = int(cleaned["was_low_coverage"].sum())
    n_imputed = int(cleaned["was_imputed"].sum())
    n_after_missing = int(cleaned["energy"].isna().sum())

    cols = [
        "timestamp", "floor_id", "energy", "energy_raw",
        "was_outlier", "was_low_coverage", "was_imputed",
        "pct_minutes_available", "temperature", "humidity", "lux",
    ]
    cleaned = cleaned[cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)

    per_floor = cleaned.groupby("floor_id").agg(
        outliers=("was_outlier", "sum"),
        low_coverage=("was_low_coverage", "sum"),
        imputed=("was_imputed", "sum"),
        remaining_nan=("energy", lambda s: s.isna().sum()),
        mean_energy_raw=("energy_raw", "mean"),
        mean_energy_cleaned=("energy", "mean"),
    ).round(2)

    report_lines = [
        "# Data Cleaning Report",
        "",
        "Generated by `src/clean.py` from `data/processed/combined_hourly.csv`, "
        "writing `data/processed/cleaned_hourly.csv`. See that script's docstring "
        "for the full justification of each rule.",
        "",
        "## Rules applied",
        "",
        f"1. Hours with <{LOW_COVERAGE_PCT:.0f}% of minutes present were treated as "
        "unreliable and blanked before imputation.",
        f"2. Per-floor outliers: energy outside median +/- {IQR_MULTIPLIER:.0f}xIQR "
        "(wide 'extreme outlier' bound, chosen so real demand peaks survive).",
        f"3. Blanked/outlier hours imputed via linear interpolation (gaps <= "
        f"{INTERP_LIMIT_HOURS}h) then seasonal-naive fill (same floor, hour-of-day, "
        "day-of-week median).",
        f"4. Sensor columns (temperature/humidity/lux) got light interpolation only "
        f"(gaps <= {SENSOR_INTERP_LIMIT_HOURS}h); longer gaps and Floor 1 (no sensors "
        "at all) are left NaN for feature engineering to handle explicitly.",
        "",
        "## Overall",
        "",
        f"- energy NaN before cleaning: {n_before_missing:,} "
        f"({n_before_missing / len(df) * 100:.2f}%)",
        f"- energy NaN after cleaning: {n_after_missing:,}",
        f"- Hours flagged as outliers (>{IQR_MULTIPLIER:.0f}xIQR): {n_outliers:,}",
        f"- Hours flagged as low-coverage (<{LOW_COVERAGE_PCT:.0f}% minutes): {n_low_coverage:,}",
        f"- Total hours imputed (missing + outlier + low-coverage, minus overlap): {n_imputed:,} "
        f"({n_imputed / len(cleaned) * 100:.2f}%)",
        "",
        "## Per floor",
        "",
        per_floor.to_markdown(),
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Before: {n_before_missing:,} NaN energy rows")
    print(f"Outliers flagged: {n_outliers:,}, low-coverage flagged: {n_low_coverage:,}")
    print(f"Total imputed: {n_imputed:,} ({n_imputed/len(cleaned)*100:.2f}%)")
    print(f"After: {n_after_missing:,} NaN energy rows remaining")
    print(f"\nWrote {OUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
