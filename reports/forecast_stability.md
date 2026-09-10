# Forecast Stability Validation

Checks the winning model's 2019 performance broken down by floor and by month, not just the aggregate score in `model_training.md` -- an aggregate can hide a floor or period the model handles badly.

## Overall (reference)

MAE 4.763, RMSE 9.807, WAPE 14.89%, R2 0.949 (n=59,630)

## Per floor

|   floor_id |    n |    MAE |   RMSE |   MAPE_% |   WAPE_% |    R2 |
|-----------:|-----:|-------:|-------:|---------:|---------:|------:|
|          1 | 8760 | 10.829 | 15.913 |   13.273 |   10.617 | 0.864 |
|          2 | 8760 |  4.711 | 10.09  |   26.286 |   17.12  | 0.906 |
|          3 | 8760 |  2.812 |  5.686 |   41.227 |   19.368 | 0.926 |
|          4 | 8760 |  3.53  |  7.572 |  149.95  |   23.53  | 0.882 |
|          5 | 8760 |  2.698 |  5.544 |   39.221 |   21.862 | 0.897 |
|          6 | 7070 |  3.213 |  6.982 |  677.466 |   14.595 | 0.952 |
|          7 | 8760 |  5.248 | 11.828 |   79.413 |   18.322 | 0.928 |

**Floors with MAE > 1.5x overall (7.14): [1]**

  - Floor 1: flagged on raw MAE, but its WAPE (10.62%) is actually *at or below* the overall WAPE (14.89%). This floor runs at a much higher baseline scale (see `eda_findings.md` -- Floor 1's mean energy is ~4-5x the other floors), so a larger absolute MAE there is expected and is not evidence of worse relative forecast quality; WAPE, which normalizes by scale, says this floor is forecast proportionally *better* than average, not worse.

## Per month (2019)

| month   |    n |   MAE |   RMSE |   MAPE_% |   WAPE_% |    R2 |
|:--------|-----:|------:|-------:|---------:|---------:|------:|
| 2019-01 | 5208 | 3.436 |  8.447 |   25.47  |   11.607 | 0.958 |
| 2019-02 | 4704 | 3.597 |  8.582 |   37.237 |   12.288 | 0.96  |
| 2019-03 | 5208 | 4.13  |  8.241 |   44.02  |   12.998 | 0.964 |
| 2019-04 | 5040 | 4.938 | 10.261 |   56.193 |   15.406 | 0.949 |
| 2019-05 | 5208 | 5.828 | 12.228 |   47.396 |   16.401 | 0.934 |
| 2019-06 | 5040 | 5.098 | 10.728 |   34.801 |   15.255 | 0.946 |
| 2019-07 | 5208 | 5.264 | 10.543 |   35.825 |   16.176 | 0.944 |
| 2019-08 | 5208 | 5.177 | 10.561 |   35.002 |   15.623 | 0.944 |
| 2019-09 | 5040 | 5.179 |  9.49  | 1137.11  |   16.155 | 0.944 |
| 2019-10 | 4982 | 4.761 |  9.297 |   34.217 |   14.76  | 0.954 |
| 2019-11 | 4320 | 4.747 |  8.827 |   33.926 |   14.517 | 0.956 |
| 2019-12 | 4464 | 4.939 |  9.382 |   42.721 |   16.986 | 0.944 |

No month exceeds 1.5x the overall MAE.

## Notes

- Floor 6 has no rows from November/December 2019 onward (its raw file truncates 2019-10-22, see `data_inspection.md`) -- those columns are blank in the per-floor x month heatmap, not zero error. Don't read the blank cells as 'perfect' or 'no data issue' -- they mean 'floor absent that month'.
- MAPE is reported for consistency with the brief but is unstable here because floors 2-7 have long near-zero-consumption stretches (overnight/weekend, see `eda_findings.md`) where tiny absolute errors become huge percentages. WAPE (sum of absolute errors over sum of actuals) is the more trustworthy aggregate percentage figure for this dataset and is what step 11's overall read should lean on.

Figures: `10_mae_floor_month_heatmap.png`, `11_mae_per_floor.png`.
