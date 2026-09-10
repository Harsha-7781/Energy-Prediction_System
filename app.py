"""
Step 15 - Streamlit UI.

Lets a user pick a floor and date range and see: the forecast (actual vs
predicted), the SHAP explanation for a specific prediction in that
range, and plain-language, SHAP-backed optimization suggestions.

Run with: streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

from src.eval_utils import compute_metrics

DATA_PATH = Path("data/processed/features_hourly.csv")
MODEL_PATH = Path("models/energy_model_latest.joblib")

HIGH_DEMAND_HOURS = range(8, 17)  # from reports/shap_explainability.md, floors 2-7

st.set_page_config(page_title="Smart Building Energy Forecast", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(bundle["model"])
    return bundle, explainer


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


@st.cache_data
def score_all(_model, feature_cols):
    df = load_data()
    df = df.copy()
    df["predicted"] = _model.predict(df[feature_cols])
    return df


def feature_contribution_chart(row: pd.Series, shap_row: np.ndarray, feature_cols: list, base_value: float):
    contrib = pd.Series(shap_row, index=feature_cols).sort_values(key=np.abs)
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in contrib.values]
    fig = go.Figure(go.Bar(
        x=contrib.values, y=contrib.index, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in contrib.values], textposition="outside",
    ))
    fig.update_layout(
        title=f"Why this prediction? (base rate {base_value:.1f} kW -> feature effects, right = increases prediction)",
        xaxis_title="SHAP contribution (kW)", height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def build_suggestions(floor_id: int, hour: int, top_features: pd.Series, has_sensor_data: int) -> list:
    suggestions = []
    if floor_id == 1:
        suggestions.append(
            "**Floor 1 does not follow the standard office-hours pattern** seen on "
            "every other floor -- its load peaks in the evening rather than midday, "
            "and its weekday/weekend consumption is nearly identical. Standard "
            "'reduce daytime AC load' advice is unlikely to apply here; the "
            "recommended next step is investigating what specifically runs on "
            "this floor in the evening (see `reports/shap_explainability.md`)."
        )
    else:
        if hour in HIGH_DEMAND_HOURS:
            suggestions.append(
                f"This hour ({hour}:00) falls inside this floor group's "
                f"identified high-demand window ({HIGH_DEMAND_HOURS.start}:00-"
                f"{HIGH_DEMAND_HOURS.stop - 1}:00). This is where load-shifting "
                "flexible equipment or pre-cooling ahead of the window has the "
                "largest modeled impact."
            )
        else:
            suggestions.append(
                f"This hour ({hour}:00) is outside the identified high-demand "
                "window for this floor group -- consumption here is already "
                "close to baseline, so there is limited additional savings "
                "opportunity at this specific time."
            )

    top_feat = top_features.index[0]
    if top_feat in ("lag_1h", "lag_24h", "lag_168h"):
        suggestions.append(
            f"The single largest driver of this specific prediction is **{top_feat}** "
            "(recent/historical load at this same slot) -- this prediction is "
            "mostly a continuation of this floor's own recent pattern rather "
            "than an unusual event."
        )
    elif top_feat == "temperature" and has_sensor_data:
        suggestions.append(
            "Temperature is the top driver for this prediction, but remember "
            "this dataset's sensor measures **indoor** room temperature, which "
            "tends to read *lower* while AC is running hardest -- don't read "
            "this as a simple 'raise the setpoint' signal without outdoor "
            "weather data to confirm the direction (see `reports/shap_explainability.md`)."
        )

    if not has_sensor_data:
        suggestions.append(
            "This floor has no working temperature/humidity/light sensors in "
            "the source data (`has_sensor_data=0`) -- its environmental "
            "features are filled with a fallback value, not a real reading."
        )
    return suggestions


def main():
    st.title("Smart Building Energy Forecast")
    st.caption(
        "CU-BEMS hourly per-floor energy forecasting, RandomForest + SHAP. "
        "Select a floor and date range to see the forecast, why the model "
        "predicts what it does for a specific hour, and what that implies "
        "for energy savings."
    )

    bundle, explainer = load_model()
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    df = score_all(model, feature_cols)

    floors = sorted(df.floor_id.unique())
    min_date, max_date = df.timestamp.min().date(), df.timestamp.max().date()

    with st.sidebar:
        st.header("Selection")
        floor = st.selectbox("Floor", floors, index=1, format_func=lambda f: f"Floor {f}")
        default_start = pd.Timestamp("2019-06-01").date()
        default_end = pd.Timestamp("2019-06-08").date()
        date_range = st.date_input(
            "Date range", value=(default_start, default_end),
            min_value=min_date, max_value=max_date,
        )
        st.caption(
            f"Data available: {min_date} to {max_date}. "
            "2018 = training period, 2019 = held-out test period."
        )

    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.warning("Select a start and end date.")
        st.stop()
    start_date, end_date = date_range

    mask = (
        (df.floor_id == floor)
        & (df.timestamp.dt.date >= start_date)
        & (df.timestamp.dt.date <= end_date)
    )
    sub = df.loc[mask].sort_values("timestamp").reset_index(drop=True)

    if sub.empty:
        st.warning(
            "No data for this floor/date range. "
            + ("Floor 6's 2019 data ends 2019-10-22 -- try an earlier range." if floor == 6 else "")
        )
        st.stop()

    st.subheader(f"Forecast: Floor {floor}, {start_date} to {end_date}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.energy, name="Actual", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=sub.timestamp, y=sub.predicted, name="Predicted", line=dict(color="#ff7f0e", dash="dot")))
    fig.update_layout(
        xaxis_title="Timestamp", yaxis_title="Energy (kW)",
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = compute_metrics(sub["energy"], sub["predicted"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['MAE']:.2f} kW")
    c2.metric("RMSE", f"{metrics['RMSE']:.2f} kW")
    c3.metric("WAPE", f"{metrics['WAPE_%']:.1f}%")
    c4.metric("R2", f"{metrics['R2']:.3f}" if not np.isnan(metrics["R2"]) else "n/a")

    st.divider()
    st.subheader("Explain a specific prediction")

    idx = st.select_slider(
        "Hour to explain",
        options=list(range(len(sub))),
        value=int(sub["predicted"].idxmax()),
        format_func=lambda i: sub.loc[i, "timestamp"].strftime("%Y-%m-%d %H:%M"),
    )
    row = sub.loc[idx]
    X_row = row[feature_cols].to_frame().T.astype(float)

    shap_row = explainer.shap_values(X_row)[0]
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(
            feature_contribution_chart(row, shap_row, feature_cols, base_value),
            use_container_width=True,
        )
    with col_b:
        st.metric("Actual energy", f"{row['energy']:.1f} kW")
        st.metric("Predicted energy", f"{row['predicted']:.1f} kW")
        st.caption(f"Model base rate (average prediction): {base_value:.1f} kW")

        st.markdown("**Optimization suggestions**")
        top_features = pd.Series(shap_row, index=feature_cols).abs().sort_values(ascending=False)
        for s in build_suggestions(int(floor), int(row["hour"]), top_features, int(row["has_sensor_data"])):
            st.markdown(f"- {s}")

    with st.expander("Model performance and background (full 2019 test set)"):
        st.markdown(
            "- Overall test-set MAE 4.76 kW, RMSE 9.81 kW, R2 0.949, WAPE 15.4% "
            "(`reports/model_evaluation.md`).\n"
            "- Floors 2-7 are strongly occupancy-driven (weekday load 3.75x-9.85x "
            "weekend); Floor 1 runs a fundamentally different, evening-peaked "
            "load with almost no weekday/weekend difference "
            "(`reports/consumption_trends.md`, `reports/shap_explainability.md`).\n"
            "- `lag_168h` (same hour, one week ago) and `lag_1h` dominate the "
            "model's predictions -- consumption is driven primarily by each "
            "floor's own repeating weekly routine.\n"
            "- Data quality notes: 2018 training data covers July-December "
            "only; Floor 6's 2019 data ends October 22 "
            "(`reports/data_inspection.md`)."
        )


if __name__ == "__main__":
    main()
