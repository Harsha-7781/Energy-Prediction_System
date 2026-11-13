"""
Resample raw minute-level per-floor CSVs to hourly and combine into a unified dataset.

Resamples each floor's raw minute-level CSV to hourly BEFORE merging
(per the confirmed data strategy), then concatenates all 7 floors x 2
years into one dataframe: timestamp, floor_id, energy, + usable
per-floor environmental features.

This is a pre-cleaning intermediate artifact. Outlier/missing-value
handling (informed by reports/data_inspection.md) happens in step 5, not
here -- this step only aggregates what is actually present in the raw
files.
"""
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/combined_hourly.csv")

FLOOR_YEAR_RE = re.compile(r"(\d{4})Floor(\d)\.csv$")


def resample_floor(path: Path) -> pd.DataFrame:
    m = FLOOR_YEAR_RE.search(path.name)
    year, floor = int(m.group(1)), int(m.group(2))

    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col)

    kw_cols = [c for c in df.columns if c.endswith("(kW)")]
    temp_cols = [c for c in df.columns if c.endswith("S1(degC)")]
    rh_cols = [c for c in df.columns if c.endswith("S1(RH%)")]
    lux_cols = [c for c in df.columns if c.endswith("S1(lux)")]

    # Per-minute floor totals/averages, before resampling.
    minute = pd.DataFrame(index=df.index)
    # min_count=1: an all-NaN minute across every kW column stays NaN
    # (genuinely no reading) rather than silently becoming 0.
    minute["energy"] = df[kw_cols].sum(axis=1, min_count=1)
    if temp_cols:
        minute["temperature"] = df[temp_cols].mean(axis=1)
    if rh_cols:
        minute["humidity"] = df[rh_cols].mean(axis=1)
    if lux_cols:
        minute["lux"] = df[lux_cols].mean(axis=1)

    hourly = minute.resample("1h").mean()
    # QC signal for the cleaning step: fraction of the 60 expected minute
    # readings actually present for that hour's energy figure.
    hourly["minutes_available"] = minute["energy"].resample("1h").count()
    hourly["pct_minutes_available"] = (hourly["minutes_available"] / 60 * 100).round(1)
    hourly = hourly.drop(columns="minutes_available")

    hourly = hourly.reset_index().rename(columns={date_col: "timestamp"})
    hourly.insert(1, "floor_id", floor)
    hourly.insert(0, "source_year", year)  # convenience column; timestamp already encodes this
    return hourly


def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    parts = []
    for f in files:
        print(f"Resampling {f.name} -> hourly ...")
        parts.append(resample_floor(f))

    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop(columns="source_year").sort_values(["floor_id", "timestamp"])
    combined = combined.reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nCombined shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print(f"Timestamp range: {combined['timestamp'].min()} -> {combined['timestamp'].max()}")
    print("\nRows per floor:")
    print(combined.groupby("floor_id").size())
    print("\nEnergy NaN count per floor (fully-missing hours):")
    print(combined.groupby("floor_id")["energy"].apply(lambda s: s.isna().sum()))
    print(f"\nWrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
