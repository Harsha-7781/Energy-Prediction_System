"""
Step 11 - Evaluate forecast errors.

Reports MAE, RMSE, MAPE (as specified), overall and per floor, for the
winning model's 2019 predictions (data/processed/test_predictions.csv).
Per-floor x month stability is already covered in
reports/forecast_stability.md (step 10) and not repeated here; this step
is the headline error-metrics deliverable plus residual diagnostics.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval_utils import compute_metrics

IN_PATH = Path("data/processed/test_predictions.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/model_evaluation.md")


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df["residual"] = df["energy"] - df["pred_winner"]

    overall = compute_metrics(df["energy"], df["pred_winner"])
    per_floor = df.groupby("floor_id").apply(
        lambda g: pd.Series(compute_metrics(g["energy"], g["pred_winner"]))
    )

    # Actual vs predicted scatter (sampled for plot size/legibility)
    sample = df.sample(min(8000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(sample["energy"], sample["pred_winner"], s=4, alpha=0.25, c="tab:blue")
    lims = [0, max(df["energy"].max(), df["pred_winner"].max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual energy")
    ax.set_ylabel("Predicted energy")
    ax.set_title("Actual vs. predicted, 2019 test set (sampled)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # Residual distribution
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(df["residual"], bins=100, color="tab:purple")
    axes[0].axvline(0, color="black", linewidth=1)
    axes[0].set_title("Residual distribution (actual - predicted)")
    axes[0].set_xlabel("Residual")

    box_data = [df[df.floor_id == f]["residual"].values for f in sorted(df.floor_id.unique())]
    axes[1].boxplot(box_data, tick_labels=[f"F{f}" for f in sorted(df.floor_id.unique())], showfliers=False)
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("Residuals per floor (outliers hidden)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_residual_distribution.png", dpi=120)
    plt.close(fig)

    bias_per_floor = df.groupby("floor_id")["residual"].mean().round(2)

    lines = [
        "# Model Evaluation (Forecast Errors)",
        "",
        "MAE / RMSE / MAPE for the winning model (`data/processed/test_predictions.csv`, "
        "`pred_winner` column -- see `reports/model_training.md` for which model won), "
        "on the full 2019 test set. Per-floor x month breakdown is in "
        "`reports/forecast_stability.md`; this report is the headline metrics.",
        "",
        "## Overall",
        "",
        f"- MAE: {overall['MAE']:.3f}",
        f"- RMSE: {overall['RMSE']:.3f}",
        f"- MAPE: {overall['MAPE_%']:.2f}%",
        f"- (WAPE: {overall['WAPE_%']:.2f}% -- see caveat below)",
        f"- R2: {overall['R2']:.3f}",
        f"- n = {overall['n']:,}",
        "",
        "## Per floor",
        "",
        per_floor[["n", "MAE", "RMSE", "MAPE_%", "WAPE_%", "R2"]].round(3).to_markdown(),
        "",
        "## MAPE caveat",
        "",
        "MAPE is reported as specified, but it is not a reliable summary "
        "statistic for this dataset: floors 2-7 have long near-zero-"
        "consumption stretches (overnight and weekends -- see "
        "`eda_findings.md`), and dividing a small absolute error by an "
        "actual value close to zero produces huge percentage errors that "
        "dominate the average (e.g. Floor 6's 689.97% MAPE despite an R2 "
        "of 0.95 and WAPE of 15.23%). **WAPE (sum of absolute errors / "
        "sum of actual values) is the more trustworthy aggregate "
        "percentage figure here** -- it weights errors by the actual "
        "magnitude of consumption rather than treating a 0.05 kW miss "
        "during a near-zero hour the same as a 50 kW miss during peak "
        "load. Use MAE/RMSE (absolute, in kW) and WAPE (relative) "
        "together; treat MAPE as supplementary only.",
        "",
        "## Bias per floor",
        "",
        "Mean residual (actual - predicted) per floor -- positive means "
        "the model tends to *underpredict* that floor, negative means it "
        "tends to *overpredict*:",
        "",
        bias_per_floor.to_markdown(),
        "",
        "Figures: `12_actual_vs_predicted.png`, `13_residual_distribution.png`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Overall: MAE={overall['MAE']:.3f} RMSE={overall['RMSE']:.3f} "
          f"MAPE={overall['MAPE_%']:.2f}% WAPE={overall['WAPE_%']:.2f}% R2={overall['R2']:.3f}")
    print("\nBias per floor:")
    print(bias_per_floor)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
