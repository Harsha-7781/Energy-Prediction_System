# Model Evaluation (Forecast Errors)

MAE / RMSE / MAPE for the winning model (`data/processed/test_predictions.csv`, `pred_winner` column -- see `reports/model_training.md` for which model won), on the full 2019 test set. Per-floor x month breakdown is in `reports/forecast_stability.md`; this report is the headline metrics.

## Overall

- MAE: 4.763
- RMSE: 9.807
- MAPE: 131.65%
- (WAPE: 14.89% -- see caveat below)
- R2: 0.949
- n = 59,630

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

## MAPE caveat

MAPE is reported as specified, but it is not a reliable summary statistic for this dataset: floors 2-7 have long near-zero-consumption stretches (overnight and weekends -- see `eda_findings.md`), and dividing a small absolute error by an actual value close to zero produces huge percentage errors that dominate the average (e.g. Floor 6's 689.97% MAPE despite an R2 of 0.95 and WAPE of 15.23%). **WAPE (sum of absolute errors / sum of actual values) is the more trustworthy aggregate percentage figure here** -- it weights errors by the actual magnitude of consumption rather than treating a 0.05 kW miss during a near-zero hour the same as a 50 kW miss during peak load. Use MAE/RMSE (absolute, in kW) and WAPE (relative) together; treat MAPE as supplementary only.

## Bias per floor

Mean residual (actual - predicted) per floor -- positive means the model tends to *underpredict* that floor, negative means it tends to *overpredict*:

|   floor_id |   residual |
|-----------:|-----------:|
|          1 |      -1.03 |
|          2 |       0.23 |
|          3 |      -1.33 |
|          4 |      -1.36 |
|          5 |      -1.15 |
|          6 |      -0.53 |
|          7 |      -0.34 |

Figures: `12_actual_vs_predicted.png`, `13_residual_distribution.png`.
