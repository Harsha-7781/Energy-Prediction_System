# Model Evaluation (Forecast Errors)

MAE / RMSE / MAPE for the winning model (`data/processed/test_predictions.csv`, `pred_winner` column -- see `reports/model_training.md` for which model won), on the full 2019 test set. Per-floor x month breakdown is in `reports/forecast_stability.md`; this report is the headline metrics.

## Overall

- MAE: 4.957
- RMSE: 9.918
- MAPE: 149.58%
- (WAPE: 15.49% -- see caveat below)
- R2: 0.948
- n = 59,630

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

## MAPE caveat

MAPE is reported as specified, but it is not a reliable summary statistic for this dataset: floors 2-7 have long near-zero-consumption stretches (overnight and weekends -- see `eda_findings.md`), and dividing a small absolute error by an actual value close to zero produces huge percentage errors that dominate the average (e.g. Floor 6's 689.97% MAPE despite an R2 of 0.95 and WAPE of 15.23%). **WAPE (sum of absolute errors / sum of actual values) is the more trustworthy aggregate percentage figure here** -- it weights errors by the actual magnitude of consumption rather than treating a 0.05 kW miss during a near-zero hour the same as a 50 kW miss during peak load. Use MAE/RMSE (absolute, in kW) and WAPE (relative) together; treat MAPE as supplementary only.

## Bias per floor

Mean residual (actual - predicted) per floor -- positive means the model tends to *underpredict* that floor, negative means it tends to *overpredict*:

|   floor_id |   residual |
|-----------:|-----------:|
|          1 |      -1.06 |
|          2 |       0.12 |
|          3 |      -1.68 |
|          4 |      -1.86 |
|          5 |      -1.57 |
|          6 |      -0.76 |
|          7 |      -1.04 |

Figures: `12_actual_vs_predicted.png`, `13_residual_distribution.png`.
