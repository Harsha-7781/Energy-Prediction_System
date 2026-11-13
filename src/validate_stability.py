"""
Forecast stability validation across building floors and months.

Checks whether the winning model's 2019 test performance is consistent
across individual floors and across months, rather than reporting one
aggregate score that could hide a floor or period where the model is
much worse. Uses data/processed/test_predictions.csv (written by
src/train.py) and the winning model's predictions (`pred_winner`).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval_utils import compute_metrics

IN_PATH = Path("data/processed/test_predictions.csv")
FIG_DIR = Path("reports/figures")
REPORT_PATH = Path("reports/forecast_stability.md")


def main():
    df = pd.read_csv(IN_PATH, parse_dates=["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)

    overall = compute_metrics(df["energy"], df["pred_winner"])

    per_floor = df.groupby("floor_id").apply(
        lambda g: pd.Series(compute_metrics(g["energy"], g["pred_winner"]))
    )
    per_month = df.groupby("month").apply(
        lambda g: pd.Series(compute_metrics(g["energy"], g["pred_winner"]))
    )
    per_floor_month = df.groupby(["floor_id", "month"]).apply(
        lambda g: pd.Series(compute_metrics(g["energy"], g["pred_winner"]))
    )["MAE"].unstack("month")

    # Flag floors/months whose MAE is notably worse than the overall MAE
    mae_threshold = overall["MAE"] * 1.5
    unstable_floors = per_floor[per_floor["MAE"] > mae_threshold]
    unstable_months = per_month[per_month["MAE"] > mae_threshold]

    # Heatmap of per-floor-per-month MAE
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(per_floor_month.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(per_floor_month.index)))
    ax.set_yticklabels([f"Floor {f}" for f in per_floor_month.index])
    ax.set_xticks(range(len(per_floor_month.columns)))
    ax.set_xticklabels(per_floor_month.columns, rotation=45, ha="right")
    ax.set_title("MAE per floor x month, 2019 test set (blank = no data that month)")
    fig.colorbar(im, ax=ax, label="MAE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_mae_floor_month_heatmap.png", dpi=120)
    plt.close(fig)

    # Bar chart of per-floor MAE vs overall
    fig, ax = plt.subplots(figsize=(9, 5))
    per_floor["MAE"].plot(kind="bar", ax=ax, color="tab:blue")
    ax.axhline(overall["MAE"], color="black", linestyle="--", label=f"Overall MAE ({overall['MAE']:.2f})")
    ax.axhline(mae_threshold, color="red", linestyle=":", label=f"1.5x overall ({mae_threshold:.2f})")
    ax.set_ylabel("MAE")
    ax.set_xlabel("Floor")
    ax.set_title("Per-floor MAE, 2019 test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_mae_per_floor.png", dpi=120)
    plt.close(fig)

    lines = [
        "# Forecast Stability Validation",
        "",
        "Checks the winning model's 2019 performance broken down by floor "
        "and by month, not just the aggregate score in `model_training.md` "
        "-- an aggregate can hide a floor or period the model handles badly.",
        "",
        f"## Overall (reference)",
        "",
        f"MAE {overall['MAE']:.3f}, RMSE {overall['RMSE']:.3f}, "
        f"WAPE {overall['WAPE_%']:.2f}%, R2 {overall['R2']:.3f} (n={overall['n']:,})",
        "",
        "## Per floor",
        "",
        per_floor.round(3).to_markdown(),
        "",
    ]
    if len(unstable_floors):
        lines += [
            f"**Floors with MAE > 1.5x overall ({mae_threshold:.2f}): "
            f"{list(unstable_floors.index)}**",
            "",
        ]
        for floor in unstable_floors.index:
            floor_wape = per_floor.loc[floor, "WAPE_%"]
            if floor_wape <= overall["WAPE_%"]:
                lines += [
                    f"  - Floor {floor}: flagged on raw MAE, but its WAPE "
                    f"({floor_wape:.2f}%) is actually *at or below* the "
                    f"overall WAPE ({overall['WAPE_%']:.2f}%). This floor "
                    "runs at a much higher baseline scale (see "
                    "`eda_findings.md` -- Floor 1's mean energy is ~4-5x "
                    "the other floors), so a larger absolute MAE there "
                    "is expected and is not evidence of worse relative "
                    "forecast quality; WAPE, which normalizes by scale, "
                    "says this floor is forecast proportionally *better* "
                    "than average, not worse.",
                    "",
                ]
    else:
        lines += ["No floor exceeds 1.5x the overall MAE -- performance is reasonably consistent across floors.", ""]

    lines += [
        "## Per month (2019)",
        "",
        per_month.round(3).to_markdown(),
        "",
    ]
    if len(unstable_months):
        lines += [
            f"**Months with MAE > 1.5x overall ({mae_threshold:.2f}): "
            f"{list(unstable_months.index)}**",
            "",
        ]
    else:
        lines += ["No month exceeds 1.5x the overall MAE.", ""]

    lines += [
        "## Notes",
        "",
        "- Floor 6 has no rows from November/December 2019 onward (its raw "
        "file truncates 2019-10-22, see `data_inspection.md`) -- those "
        "columns are blank in the per-floor x month heatmap, not zero "
        "error. Don't read the blank cells as 'perfect' or 'no data "
        "issue' -- they mean 'floor absent that month'.",
        "- MAPE is reported for consistency with the brief but is unstable "
        "here because floors 2-7 have long near-zero-consumption stretches "
        "(overnight/weekend, see `eda_findings.md`) where tiny absolute "
        "errors become huge percentages. WAPE (sum of absolute errors "
        "over sum of actuals) is the more trustworthy aggregate percentage "
        "figure for this dataset and is what step 11's overall read should "
        "lean on.",
        "",
        "Figures: `10_mae_floor_month_heatmap.png`, `11_mae_per_floor.png`.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"Overall: {overall}")
    print("\nPer floor:")
    print(per_floor.round(2))
    print("\nPer month:")
    print(per_month.round(2))
    print(f"\nUnstable floors (>1.5x MAE): {list(unstable_floors.index)}")
    print(f"Unstable months (>1.5x MAE): {list(unstable_months.index)}")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
