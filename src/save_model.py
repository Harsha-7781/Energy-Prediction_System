"""
Step 14 - Save the model.

Retrains the winning model (per reports/model_training.md) on all 2018
data with the exact hyperparameters from src/train.py, and saves it with
a clear, versioned filename for the Streamlit app (step 15) to load.
Retraining here (rather than just copying models/interim_winner.joblib)
keeps this step self-contained and reproducible from source, independent
of whatever earlier steps happened to leave on disk.
"""
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

TRAIN_PATH = Path("data/processed/train.csv")
MODEL_DIR = Path("models")

FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "floor_id",
    "lag_1h", "lag_24h", "lag_168h", "roll_mean_3h", "roll_mean_24h",
    "temperature", "humidity", "lux", "has_sensor_data",
]
TARGET_COL = "energy"

VERSION = "v1"


def main():
    train_df = pd.read_csv(TRAIN_PATH, parse_dates=["timestamp"])
    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]

    # Must match src/train.py's chosen hyperparameters -- see that
    # script's comment for the size/accuracy justification.
    model = RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=10,
        n_jobs=-1, random_state=42,
    )
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    filename = f"energy_model_randomforest_{VERSION}_{date.today().isoformat()}.joblib"
    out_path = MODEL_DIR / filename

    bundle = {
        "model": model,
        "model_type": "RandomForestRegressor",
        "version": VERSION,
        "trained_on": "2018 (all floors)",
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "sklearn_params": model.get_params(),
    }
    joblib.dump(bundle, out_path)

    # Also write a stable, unversioned symlink-equivalent (plain copy) so
    # the Streamlit app always has one fixed path to load regardless of
    # the date-stamped filename.
    latest_path = MODEL_DIR / "energy_model_latest.joblib"
    joblib.dump(bundle, latest_path)

    print(f"Saved {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Saved {latest_path} (stable path for the Streamlit app)")


if __name__ == "__main__":
    main()
