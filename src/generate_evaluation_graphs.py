"""
Generate publication-quality evaluation metric graphs.
Produces comprehensive visual summaries of model evaluation:
- 18_model_comparison_metrics.png: Random Forest vs Gradient Boosting metrics.
- 19_floor_evaluation_metrics_summary.png: MAE, RMSE, R2, and WAPE across floors.
- 20_actual_vs_predicted_timeseries_tracking.png: High-resolution actual vs forecast tracking.
- 21_residual_diagnostics_by_hour.png: Error patterns by hour of day and scatter density.
"""
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set non-interactive backend and avoid font cache issues
os.environ["MPLCONFIGDIR"] = str(Path("reports/.mpl_cache").resolve())
Path("reports/.mpl_cache").mkdir(parents=True, exist_ok=True)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR = Path("images")
IMG_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = Path("data/processed/test_predictions.csv")
if not PRED_PATH.exists() and Path("Data/processed/test_predictions.csv").exists():
    PRED_PATH = Path("Data/processed/test_predictions.csv")


def plot_model_comparison():
    models = ["Random Forest", "Gradient Boosting"]
    mae = [4.763, 12.089]
    rmse = [9.807, 17.810]
    r2 = [0.949, 0.833]
    train_time = [2.4, 14.7]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    colors = ["#1f77b4", "#ff7f0e"]

    # MAE & RMSE
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, mae, width, label="MAE (kW)", color="#2ca02c", alpha=0.85)
    axes[0].bar(x + width/2, rmse, width, label="RMSE (kW)", color="#d62728", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontweight="bold")
    axes[0].set_ylabel("Error (kW)")
    axes[0].set_title("Forecast Error: MAE & RMSE (Lower is Better)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    for i, v in enumerate(mae):
        axes[0].text(i - width/2, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    for i, v in enumerate(rmse):
        axes[0].text(i + width/2, v + 0.3, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    # R2 Score
    bars = axes[1].bar(models, r2, color=colors, width=0.5, alpha=0.85)
    axes[1].set_ylabel("R² Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Variance Explained: R² Score (Higher is Better)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.3f}", ha="center", fontweight="bold")

    # Training Time
    bars_t = axes[2].bar(models, train_time, color=["#3498db", "#9b59b6"], width=0.5, alpha=0.85)
    axes[2].set_ylabel("Seconds")
    axes[2].set_title("Training Duration (Seconds)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars_t:
        h = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}s", ha="center", fontweight="bold")

    plt.suptitle("Model Benchmark: Random Forest vs. Gradient Boosting (2019 Test Set)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "18_model_comparison_metrics.png", dpi=150, bbox_inches="tight")
    fig.savefig(IMG_DIR / "18_model_comparison_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 18_model_comparison_metrics.png")


def plot_floor_metrics():
    floors = [f"Floor {i}" for i in range(1, 8)]
    mae = [10.829, 4.711, 2.812, 3.530, 2.698, 3.213, 5.248]
    rmse = [15.913, 10.090, 5.686, 7.572, 5.544, 6.982, 11.828]
    wape = [10.617, 17.120, 19.368, 23.530, 21.862, 14.595, 18.322]
    r2 = [0.864, 0.906, 0.926, 0.882, 0.897, 0.952, 0.928]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. MAE
    bars1 = axes[0, 0].bar(floors, mae, color="#2980b9", alpha=0.85)
    axes[0, 0].axhline(4.763, color="#e74c3c", linestyle="--", linewidth=1.5, label="Overall MAE: 4.76 kW")
    axes[0, 0].set_title("Mean Absolute Error (MAE) per Floor", fontweight="bold")
    axes[0, 0].set_ylabel("MAE (kW)")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 0].legend()
    for bar in bars1:
        h = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, h + 0.2, f"{h:.2f}", ha="center", fontsize=8.5)

    # 2. RMSE
    bars2 = axes[0, 1].bar(floors, rmse, color="#8e44ad", alpha=0.85)
    axes[0, 1].axhline(9.807, color="#e74c3c", linestyle="--", linewidth=1.5, label="Overall RMSE: 9.81 kW")
    axes[0, 1].set_title("Root Mean Squared Error (RMSE) per Floor", fontweight="bold")
    axes[0, 1].set_ylabel("RMSE (kW)")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0, 1].legend()
    for bar in bars2:
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}", ha="center", fontsize=8.5)

    # 3. WAPE
    bars3 = axes[1, 0].bar(floors, wape, color="#16a085", alpha=0.85)
    axes[1, 0].axhline(14.89, color="#e74c3c", linestyle="--", linewidth=1.5, label="Overall WAPE: 14.89%")
    axes[1, 0].set_title("Weighted Absolute Percentage Error (WAPE) per Floor", fontweight="bold")
    axes[1, 0].set_ylabel("WAPE (%)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 0].legend()
    for bar in bars3:
        h = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.1f}%", ha="center", fontsize=8.5)

    # 4. R2
    bars4 = axes[1, 1].bar(floors, r2, color="#27ae60", alpha=0.85)
    axes[1, 1].axhline(0.949, color="#e74c3c", linestyle="--", linewidth=1.5, label="Overall R²: 0.949")
    axes[1, 1].set_title("Goodness of Fit (R²) per Floor", fontweight="bold")
    axes[1, 1].set_ylabel("R² Score")
    axes[1, 1].set_ylim(0.7, 1.02)
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 1].legend()
    for bar in bars4:
        h = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, h + 0.007, f"{h:.3f}", ha="center", fontsize=8.5)

    plt.suptitle("Detailed Model Performance Breakdown by Building Floor (2019 Test Set)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "19_floor_evaluation_metrics_summary.png", dpi=150, bbox_inches="tight")
    fig.savefig(IMG_DIR / "19_floor_evaluation_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 19_floor_evaluation_metrics_summary.png")


def plot_timeseries_tracking():
    if not PRED_PATH.exists():
        print(f"Warning: {PRED_PATH} not found, skipping timeseries tracking plot.")
        return

    df = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])
    # Sample a clean 7-day test window in June 2019
    start_date = pd.Timestamp("2019-06-03")
    end_date = pd.Timestamp("2019-06-10")

    mask_f3 = (df["floor_id"] == 3) & (df["timestamp"] >= start_date) & (df["timestamp"] < end_date)
    mask_f1 = (df["floor_id"] == 1) & (df["timestamp"] >= start_date) & (df["timestamp"] < end_date)

    sub3 = df.loc[mask_f3].sort_values("timestamp")
    sub1 = df.loc[mask_f1].sort_values("timestamp")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Floor 3 (Standard office pattern)
    axes[0].plot(sub3["timestamp"], sub3["energy"], label="Actual Demand (kW)", color="#1f77b4", linewidth=2.0)
    axes[0].plot(sub3["timestamp"], sub3["pred_winner"], label="Random Forest Forecast (kW)", color="#ff7f0e", linestyle="--", linewidth=2.0)
    axes[0].set_title("Floor 3: Standard Office Profile (Actual vs. Forecast Tracking, June 2019)", fontweight="bold")
    axes[0].set_ylabel("Power Demand (kW)")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper right")

    # Floor 1 (Evening peaked pattern)
    axes[1].plot(sub1["timestamp"], sub1["energy"], label="Actual Demand (kW)", color="#1f77b4", linewidth=2.0)
    axes[1].plot(sub1["timestamp"], sub1["pred_winner"], label="Random Forest Forecast (kW)", color="#ff7f0e", linestyle="--", linewidth=2.0)
    axes[1].set_title("Floor 1: High-Baseline & Evening-Peaked Profile (Actual vs. Forecast Tracking, June 2019)", fontweight="bold")
    axes[1].set_ylabel("Power Demand (kW)")
    axes[1].set_xlabel("Timestamp")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right")

    plt.suptitle("Model Forecast Precision: 7-Day Continuous Tracking Across Contrasting Building Floors", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "20_actual_vs_predicted_timeseries_tracking.png", dpi=150, bbox_inches="tight")
    fig.savefig(IMG_DIR / "20_actual_vs_predicted_timeseries_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 20_actual_vs_predicted_timeseries_tracking.png")


def plot_residual_diagnostics():
    if not PRED_PATH.exists():
        print(f"Warning: {PRED_PATH} not found, skipping residual diagnostics plot.")
        return

    df = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["abs_error"] = (df["energy"] - df["pred_winner"]).abs()
    df["residual"] = df["energy"] - df["pred_winner"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Error by Hour of Day
    hourly_mae = df.groupby(["hour", "floor_id"])["abs_error"].mean().unstack()
    for col in hourly_mae.columns:
        axes[0].plot(hourly_mae.index, hourly_mae[col], marker="o", markersize=4, label=f"Floor {col}")
    axes[0].set_title("Mean Absolute Error (MAE) Across Hours of Day", fontweight="bold")
    axes[0].set_xlabel("Hour of Day (00:00 - 23:00)")
    axes[0].set_ylabel("MAE (kW)")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.5)

    # Actual vs Predicted Scatter with Density
    sample = df.sample(min(12000, len(df)), random_state=42)
    max_val = max(sample["energy"].max(), sample["pred_winner"].max())
    axes[1].scatter(sample["energy"], sample["pred_winner"], alpha=0.25, s=6, c="#2980b9", edgecolors="none")
    axes[1].plot([0, max_val], [0, max_val], "r--", linewidth=2.0, label="Ideal 1:1 Parity")
    axes[1].set_title("Predicted vs. Actual Power Demand (Sampled n=12,000)", fontweight="bold")
    axes[1].set_xlabel("Observed Energy Consumption (kW)")
    axes[1].set_ylabel("Predicted Energy Consumption (kW)")
    axes[1].set_xlim(0, max_val * 1.02)
    axes[1].set_ylim(0, max_val * 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper left")

    plt.suptitle("Residual Diagnostics & Hourly Error Stability Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "21_residual_diagnostics_by_hour.png", dpi=150, bbox_inches="tight")
    fig.savefig(IMG_DIR / "21_residual_diagnostics_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 21_residual_diagnostics_by_hour.png")


def copy_all_figures_to_images():
    """Ensure all figures (EDA, decomposition, SHAP, and evaluation) are in images/."""
    for img_file in FIG_DIR.glob("*.png"):
        dest = IMG_DIR / img_file.name
        dest.write_bytes(img_file.read_bytes())
    print(f"Copied all figures from {FIG_DIR} to {IMG_DIR}")


if __name__ == "__main__":
    plot_model_comparison()
    plot_floor_metrics()
    plot_timeseries_tracking()
    plot_residual_diagnostics()
    copy_all_figures_to_images()
    print("All evaluation metric figures successfully generated!")

