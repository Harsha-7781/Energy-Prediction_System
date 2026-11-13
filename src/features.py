"""
Feature engineering pipeline: cyclical time, lags, rolling statistics, and sensors.

Builds the model-ready feature table from data/processed/cleaned_hourly.csv:
time-based features (hour, day-of-week, month, is_weekend), lag features,
rolling averages, floor_id as a (numerically-encoded) categorical feature,
and available environmental sensor readings. Writes
data/processed/features_hourly.csv.

Design decisions:

- All lag/rolling features are computed PER FLOOR on the full, chronologically
  ordered series (2018 + 2019 together) BEFORE any train/test split, using
  only past values (`.shift(1)` before rolling, so a rolling window never
  includes the row it's predicting). This means a January-2019 test row
  legitimately gets lag/rolling context from late-2018 history -- that is
  real past information, not leakage. Only the first ~168 hours of each
  floor's series (2018-07-01 onward) lack enough history for the 168h lag
  and get dropped.
- floor_id is fed to the trees as a plain integer (1-7), not one-hot
  encoded. With only 7 categories, tree ensembles (RandomForest /
  GradientBoosting) can split on it freely and effectively learn
  floor-specific behavior; one-hot would add 7 sparse columns for no real
  benefit here and would clutter the SHAP plots in step 13.
- Time features are kept as plain integers (hour, day_of_week, month), not
  cyclical sin/cos encodings -- trees don't need the smoothness cyclical
  encoding buys linear models, and plain integers are far easier to
  translate into the plain-language recommendations step 13 needs.
- Sensor features (temperature/humidity/lux) are missing at very different
  rates per floor (step 2/3 found Floor 1 has NONE at all). Remaining NaNs
  after step 5's light interpolation are filled with each floor's OWN
  2018 (training-period) median, so no test-period statistics leak into
  the fill value. Floors with no training-period sensor data at all
  (Floor 1) fall back to the global 2018 median across floors, and a
  binary `has_sensor_data` flag marks which floor actually has real
  sensor readings vs. a fallback constant -- without this flag the model
  could not tell a real reading from an imputed placeholder.
- `energy_raw` / `was_imputed` / `was_outlier` are carried through as
  metadata (useful for step 10/11 QC breakdowns) but are NOT model
  features -- whether a row was imputed is not knowable at prediction
  time for genuinely new data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

IN_PATH = Path("data/processed/cleaned_hourly.csv")
OUT_PATH = Path("data/processed/features_hourly.csv")
REPORT_PATH = Path("reports/feature_engineering.md")

LAGS = [1, 24, 168]  # 1h, 1 day, 1 week
ROLLING_WINDOWS = [3, 24]  # past 3h, past 24h


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_and_rolling(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("timestamp").reset_index(drop=True)
    for lag in LAGS:
        group[f"lag_{lag}h"] = group["energy"].shift(lag)
    past = group["energy"].shift(1)  # never include the current hour itself
    for window in ROLLING_WINDOWS:
        group[f"roll_mean_{window}h"] = past.rolling(window).mean()
    return group


def impute_sensors(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    sensor_cols = ["temperature", "humidity", "lux"]
    has_sensor_data = df.groupby("floor_id")[sensor_cols[0]].transform(
        lambda s: s.notna().any()
    )
    df["has_sensor_data"] = has_sensor_data.astype(int)

    global_train_medians = df.loc[train_mask, sensor_cols].median()
    for col in sensor_cols:
        floor_train_medians = df.loc[train_mask].groupby("floor_id")[col].median()
        fill_values = df["floor_id"].map(floor_train_medians)
        fill_values = fill_values.fillna(global_train_medians[col])
        df[col] = df[col].fillna(fill_values)
    return df


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df = add_time_features(df)

    parts = [add_lag_and_rolling(df[df.floor_id == f].copy()) for f in sorted(df.floor_id.unique())]
    df = pd.concat(parts, ignore_index=True)

    train_mask = df["timestamp"].dt.year == 2018
    df = impute_sensors(df, train_mask)

    engineered_cols = [f"lag_{l}h" for l in LAGS] + [f"roll_mean_{w}h" for w in ROLLING_WINDOWS]
    n_before = len(df)
    dropped = df[engineered_cols].isna().any(axis=1)
    df_clean = df.loc[~dropped].reset_index(drop=True)
    n_dropped = int(dropped.sum())

    feature_cols = (
        ["hour", "day_of_week", "month", "is_weekend", "floor_id"]
        + engineered_cols
        + ["temperature", "humidity", "lux", "has_sensor_data"]
    )
    out_cols = ["timestamp", "floor_id", "energy"] + [c for c in feature_cols if c != "floor_id"] + [
        "energy_raw", "was_imputed", "was_outlier",
    ]
    df_clean = df_clean[out_cols]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(OUT_PATH, index=False)

    months_2018 = sorted(int(m) for m in df_clean.loc[df_clean.timestamp.dt.year == 2018, "month"].unique())
    months_2019 = sorted(int(m) for m in df_clean.loc[df_clean.timestamp.dt.year == 2019, "month"].unique())

    lines = [
        "# Feature Engineering",
        "",
        "Generated by `src/features.py` from `data/processed/cleaned_hourly.csv`, "
        "writing `data/processed/features_hourly.csv`. See that script's docstring "
        "for design rationale (why floor_id is integer-encoded, why lags/rolling "
        "are computed pre-split, sensor imputation strategy).",
        "",
        f"## Features ({len(feature_cols)})",
        "",
        f"- Time: `hour`, `day_of_week` (0=Mon), `month`, `is_weekend`",
        f"- Floor: `floor_id` (integer 1-7, fed directly to the trees)",
        f"- Lags: {', '.join(f'`lag_{l}h`' for l in LAGS)}",
        f"- Rolling means (past, non-leaking): {', '.join(f'`roll_mean_{w}h`' for w in ROLLING_WINDOWS)}",
        f"- Sensors: `temperature`, `humidity`, `lux`, `has_sensor_data` "
        "(0/1 flag -- Floor 1 has no real sensor data and is always 0)",
        "",
        "## Row count",
        "",
        f"- Before dropping warm-up rows (insufficient lag history): {n_before:,}",
        f"- Dropped (first ~{max(LAGS)}h of each floor's series, can't compute "
        f"lag_{max(LAGS)}h yet): {n_dropped:,}",
        f"- Final: {len(df_clean):,}",
        "",
        "## Important caveat for modeling",
        "",
        f"- 2018 (training) months present: {months_2018}",
        f"- 2019 (test) months present: {months_2019}",
        "- **The model never sees January-June in training** (2018 data starts "
        "July 1st -- see data_inspection.md). The `month` feature will take "
        "values at test time (1-6) that never appeared during training. Tree "
        "models can still use `hour`/`day_of_week`/lag/rolling context for "
        "those months, but any genuinely month-specific seasonal effect for "
        "H1 has to be inferred, not learned directly. This is an intentional "
        "consequence of the confirmed 2018-train/2019-test split and should "
        "be kept in mind when interpreting H1-2019 errors in step 10/11.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Rows before/after warm-up drop: {n_before:,} -> {len(df_clean):,} ({n_dropped:,} dropped)")
    print(f"2018 months present: {months_2018}")
    print(f"2019 months present: {months_2019}")
    print(f"\nWrote {OUT_PATH}")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
