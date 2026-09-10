"""
Step 8 - Train/test split.

Chronological split by year, per the confirmed data strategy: train on
all 2018 rows, test on all 2019 rows. Not a random split -- this is
intentional (full-year-ahead generalization test), so keep it this way
even where step 10/11 metrics look worse than a random split would give.

Writes data/processed/train.csv and data/processed/test.csv from
data/processed/features_hourly.csv.
"""
from pathlib import Path

import pandas as pd

IN_PATH = Path("data/processed/features_hourly.csv")
TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])

    train = df[df.timestamp.dt.year == 2018].reset_index(drop=True)
    test = df[df.timestamp.dt.year == 2019].reset_index(drop=True)

    assert train.timestamp.max() < test.timestamp.min(), (
        "Train/test overlap or ordering violation -- split is supposed to be chronological."
    )

    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print(f"Train (2018): {len(train):,} rows, {train.timestamp.min()} -> {train.timestamp.max()}")
    print(f"Test  (2019): {len(test):,} rows, {test.timestamp.min()} -> {test.timestamp.max()}")
    print("\nTrain rows per floor:")
    print(train.groupby("floor_id").size())
    print("\nTest rows per floor:")
    print(test.groupby("floor_id").size())
    print(f"\nWrote {TRAIN_PATH}")
    print(f"Wrote {TEST_PATH}")


if __name__ == "__main__":
    main()
