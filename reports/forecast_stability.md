# Forecast Stability Validation

Checks the winning model's 2019 performance broken down by floor and by month, not just the aggregate score in `model_training.md` -- an aggregate can hide a floor or period the model handles badly.

## Overall (reference)

MAE 4.957, RMSE 9.918, WAPE 15.49%, R2 0.948 (n=59,630)

## Per floor

|   floor_id |    n |    MAE |   RMSE |   MAPE_% |   WAPE_% |    R2 |
|-----------:|-----:|-------:|-------:|---------:|---------:|------:|
|          1 | 8760 | 10.703 | 15.761 |   12.962 |   10.494 | 0.866 |
|          2 | 8760 |  4.771 |  9.699 |   28.129 |   17.335 | 0.913 |
|          3 | 8760 |  3.03  |  5.981 |   69.31  |   20.873 | 0.918 |
|          4 | 8760 |  3.794 |  8.552 |  184.398 |   25.291 | 0.85  |
|          5 | 8760 |  3.062 |  5.998 |   61.285 |   24.809 | 0.88  |
|          6 | 7070 |  3.353 |  7.062 |  689.973 |   15.227 | 0.951 |
|          7 | 8760 |  5.673 | 11.914 |  105.283 |   19.808 | 0.927 |

**Floors with MAE > 1.5x overall (7.43): [1]**

  - Floor 1: flagged on raw MAE, but its WAPE (10.49%) is actually *at or below* the overall WAPE (15.49%). This floor runs at a much higher baseline scale (see `eda_findings.md` -- Floor 1's mean energy is ~4-5x the other floors), so a larger absolute MAE there is expected and is not evidence of worse relative forecast quality; WAPE, which normalizes by scale, says this floor is forecast proportionally *better* than average, not worse.

## Per month (2019)

| month   |    n |   MAE |   RMSE |   MAPE_% |   WAPE_% |    R2 |
|:--------|-----:|------:|-------:|---------:|---------:|------:|
| 2019-01 | 5208 | 3.259 |  7.874 |   24.613 |   11.008 | 0.964 |
| 2019-02 | 4704 | 3.543 |  8.223 |   44.427 |   12.103 | 0.963 |
| 2019-03 | 5208 | 4.432 |  8.709 |   62.374 |   13.947 | 0.959 |
| 2019-04 | 5040 | 5.147 | 10.288 |   79.045 |   16.061 | 0.948 |
| 2019-05 | 5208 | 5.948 | 12.015 |   56.95  |   16.736 | 0.937 |
| 2019-06 | 5040 | 5.343 | 10.849 |   45.547 |   15.987 | 0.944 |
| 2019-07 | 5208 | 5.527 | 10.796 |   49.347 |   16.984 | 0.941 |
| 2019-08 | 5208 | 5.382 | 10.725 |   47.266 |   16.242 | 0.942 |
| 2019-09 | 5040 | 5.286 |  9.597 | 1190.66  |   16.491 | 0.942 |
| 2019-10 | 4982 | 4.998 |  9.525 |   50.793 |   15.492 | 0.951 |
| 2019-11 | 4320 | 5.017 |  9.232 |   44.743 |   15.344 | 0.951 |
| 2019-12 | 4464 | 5.593 | 10.157 |   85.734 |   19.234 | 0.935 |

No month exceeds 1.5x the overall MAE.

## Notes

- Floor 6 has no rows from November/December 2019 onward (its raw file truncates 2019-10-22, see `data_inspection.md`) -- those columns are blank in the per-floor x month heatmap, not zero error. Don't read the blank cells as 'perfect' or 'no data issue' -- they mean 'floor absent that month'.
- MAPE is reported for consistency with the brief but is unstable here because floors 2-7 have long near-zero-consumption stretches (overnight/weekend, see `eda_findings.md`) where tiny absolute errors become huge percentages. WAPE (sum of absolute errors over sum of actuals) is the more trustworthy aggregate percentage figure for this dataset and is what step 11's overall read should lean on.

Figures: `10_mae_floor_month_heatmap.png`, `11_mae_per_floor.png`.
