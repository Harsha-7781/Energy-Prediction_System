# Energy Consumption Prediction for Smart Buildings

Student deliverable: a Streamlit app that forecasts hourly per-floor energy
consumption for a smart building and explains the prediction (SHAP-based
XAI) with plain-language optimization suggestions.

## Dataset

CU-BEMS (Chulalongkorn University Building Energy Management System).
14 raw CSVs in `data/raw/` — 7 floors x 2 years (2018, 2019), one file per
floor per year (`2018Floor1.csv` ... `2019Floor7.csv`).

Confirmed structure (inspected, not assumed):
- Near-minute sampling frequency (irregular — see data inspection report).
- Each file has a `Date` column (`YYYY-MM-DD HH:MM:SS`) plus per-zone,
  per-device power columns named `z{zone}_{device}(kW)` (e.g. `z1_AC1(kW)`,
  `z2_Light(kW)`, `z3_Plug(kW)`), and some floors additionally have
  per-zone sensor columns `z{zone}_S1(degC)`, `z{zone}_S1(RH%)`,
  `z{zone}_S1(lux)` for temperature, humidity, and illuminance.
- **Schema is NOT consistent across floors.** Column count varies:
  Floor 1 = 12 cols (no sensor columns, 4 zones), Floor 2 = 37 cols
  (14 AC units in zone 2, has sensors), Floors 3-7 = 30 cols (5 zones,
  has sensors). Each floor's own 2018 vs 2019 file uses the same schema.
  Any merge/combine logic must handle this per-floor, not assume a common
  column set.
- Known CU-BEMS data quality issues to expect and document (not silently
  patch): sensor dropout gaps, occasional negative/implausible spikes,
  missing S1 sensor readings (blank fields), and multi-day outages.

## Data strategy (do not deviate without asking)

1. Inspect each of the 14 files individually first (schema, shape, date
   range, missing-data patterns, fault gaps) — done in
   `reports/data_inspection.md`.
2. Per floor: sum all `*_kW` columns (across zones/devices) into a single
   `energy` value per timestamp, then resample to hourly (mean or sum of
   the minute-level energy within the hour — decide and justify in the
   cleaning step) BEFORE combining floors.
3. Combine all 7 floors x 2 years into one dataframe:
   `timestamp, floor_id, energy, <other usable features>`.
4. Train ONE model on the combined data with `floor_id` as a categorical
   feature — not 7 separate per-floor models.
5. Split chronologically by year: train on all 2018 rows, test/validate on
   all 2019 rows. This is intentional (tests generalization across a full
   year gap) — keep this split even if in-sample metrics look worse than a
   random split would give.

## Folder structure

```
Energy_Prediction/
  CLAUDE.md
  requirements.txt
  problem_statement.md
  data/
    raw/            14 source CSVs (gitignored, not committed)
    interim/         per-floor hourly-resampled files (gitignored)
    processed/        combined_hourly.csv + cleaned/feature-engineered versions (gitignored)
  src/               reusable Python modules (ingestion, cleaning, features, train, evaluate, shap_explain)
  notebooks/         exploratory notebooks, if any
  reports/
    figures/         saved EDA / evaluation plots
    data_inspection.md
  models/            joblib-serialized trained models (gitignored, versioned filenames)
  app.py             Streamlit UI (final step)
  .streamlit/
```

## Environment

- Python 3.13.5 (project asked for 3.11; 3.11 was not available on this
  machine — using 3.13 since no 3.11-only feature/library constraint
  applies here. Flag if this becomes a problem for a grading environment).
- Virtualenv at `.venv/` (gitignored). Activate: `.venv/Scripts/activate`
  (PowerShell: `.venv\Scripts\Activate.ps1`).
- Installed versions (see `requirements.txt`):
  pandas 3.0.5, numpy 2.5.3, scikit-learn 1.9.0, shap 0.52.0, plotly 7.0.0,
  streamlit 1.63.0, joblib 1.6.0, matplotlib 3.11.1, statsmodels 0.15.0
  (matplotlib/statsmodels added for static EDA plots and seasonal
  decomposition, not in the original tech-stack list but needed for steps
  4 and 6).

## Workflow / build order

Building incrementally, 15 steps, committing to git after each completed
step, pausing for review every 4 steps:

1. Problem definition (`problem_statement.md`)
2. Data ingestion — inspect all 14 raw CSVs individually
3. Combine — resample to hourly per floor, merge into `data/processed/combined_hourly.csv`
4. EDA — time patterns per floor and combined, save plots
5. Data cleaning — missing values + outlier logic (justified from step 2 findings)
6. Seasonal trend decomposition
7. Feature engineering — time features, lags, rolling averages, floor_id
8. Train/test split — 2018 train / 2019 test (chronological)
9. Model training — RandomForestRegressor baseline, compare GradientBoosting
10. Validate forecast stability — per-floor and per-month breakdown
11. Evaluate forecast errors — MAE/RMSE/MAPE overall and per floor
12. Interpret consumption trends — plain-language, model-backed
13. Explainability — SHAP feature importance -> energy-saving recommendations
14. Save model — joblib, versioned filename
15. Streamlit UI — floor + date range selection, forecast, SHAP explanation, suggestions

Stop after step 15; testing/deployment handled separately by the user.
