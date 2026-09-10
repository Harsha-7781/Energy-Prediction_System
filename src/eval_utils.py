"""
Shared metric helpers for steps 10/11 so both reports compute error
figures the exact same way (no drift between "validate stability" and
"evaluate forecast errors").
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

    # Plain MAPE blows up when actual values are near zero (common here --
    # floors 2-7 sit near 0 kW overnight/weekends, see reports/eda_findings.md).
    # Report it as specified, but alongside WAPE (sum|error|/sum|actual|),
    # which stays stable near zero and is the more trustworthy aggregate
    # figure for this data -- see reports/model_evaluation.md for discussion.
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-3, None))) * 100)
    wape = float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)

    return {"n": len(y_true), "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "WAPE_%": wape, "R2": r2}
