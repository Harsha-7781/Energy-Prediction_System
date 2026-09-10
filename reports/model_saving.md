# Model Saving

`src/save_model.py` retrains the winning RandomForest configuration on
all 2018 data and saves it to `models/energy_model_randomforest_v1_<date>.joblib`,
plus a stable `models/energy_model_latest.joblib` copy the Streamlit app
loads (so the app doesn't need to know today's date).

## Hyperparameter refinement (steps 9-14)

Step 9 originally evaluated an unconstrained RandomForest
(`n_estimators=300, max_depth=None, min_samples_leaf=2`). That scored
well (MAE 4.96, R2 0.948) but serialized to **351.6 MB** and made exact
SHAP `TreeExplainer` computation (step 13) impractically slow -- a
single run on a few thousand rows did not finish in several minutes,
which would also make the live per-prediction SHAP explanation in the
Streamlit app (step 15) unusably slow.

A small hyperparameter sweep against the same 2018-train/2019-test split
found a much lighter configuration that is **not a tradeoff**:

| Config | MAE | RMSE | R2 | Size |
|---|---|---|---|---|
| n_estimators=300, max_depth=None, min_samples_leaf=2 (original) | 4.957 | 9.918 | 0.9482 | 351.6 MB |
| n_estimators=200, max_depth=18, min_samples_leaf=5 | 4.908 | 9.974 | 0.9476 | 60.3 MB |
| **n_estimators=150, max_depth=14, min_samples_leaf=10 (chosen)** | **4.763** | **9.807** | **0.9493** | **18.9 MB** |

The chosen configuration scores *slightly better* on every metric while
being ~19x smaller -- the original unconstrained forest was mildly
overfitting the ~30k-row training set (very deep trees, leaves of size
2). `src/train.py` was updated to use this configuration, and steps
9-13's reports/figures were regenerated against it so every number in
this repository is internally consistent with the model actually saved
and shipped.

## Saved bundle contents

`joblib.dump`'d dict with: `model` (the fitted `RandomForestRegressor`),
`model_type`, `version`, `trained_on`, `feature_cols` (the exact column
order the model expects), `target_col`, and `sklearn_params` (full
hyperparameter record, for reproducibility without re-reading source).
