# Model Training

Single model trained on all 2018 floors combined (`floor_id` as a feature), evaluated on all 2019 data. Both models use the same 14 features from `src/features.py`.

## Results (2019 test set, all floors combined)

| model            |    MAE |   RMSE |   MAPE_% |    R2 |   train_seconds |
|:-----------------|-------:|-------:|---------:|------:|----------------:|
| RandomForest     |  4.957 |  9.918 |  149.584 | 0.948 |             6.8 |
| GradientBoosting | 12.089 | 17.81  |  802.583 | 0.833 |            13.2 |

## RandomForest vs GradientBoosting

- MAE gap: 59.0% (RandomForest lower)
- RandomForest performs meaningfully better (>5% MAE improvement).
- **Selected for downstream steps: RandomForest**

## Feature importances

### RandomForest

|                 |   importance |
|:----------------|-------------:|
| lag_168h        |       0.6803 |
| lag_1h          |       0.2571 |
| lag_24h         |       0.0187 |
| hour            |       0.0159 |
| roll_mean_3h    |       0.0133 |
| roll_mean_24h   |       0.0048 |
| floor_id        |       0.0028 |
| day_of_week     |       0.0019 |
| temperature     |       0.0017 |
| lux             |       0.0013 |
| month           |       0.0011 |
| humidity        |       0.0006 |
| is_weekend      |       0.0005 |
| has_sensor_data |       0.0001 |

### GradientBoosting

|                 |   importance |
|:----------------|-------------:|
| lag_168h        |       0.5694 |
| lag_1h          |       0.3416 |
| lag_24h         |       0.0518 |
| hour            |       0.0174 |
| roll_mean_3h    |       0.0109 |
| roll_mean_24h   |       0.003  |
| temperature     |       0.0027 |
| lux             |       0.0013 |
| floor_id        |       0.0011 |
| day_of_week     |       0.0004 |
| month           |       0.0001 |
| humidity        |       0.0001 |
| has_sensor_data |       0.0001 |
| is_weekend      |       0      |

## Notes

- `lag_168h` (same hour, one week ago) dominates both models (~57-68%), with `lag_1h` second -- consistent with step 6's finding that the daily/weekly cycle explains most of the variance for most floors. These lag features assume the model has access to that floor's own recent actual meter readings at prediction time, which is realistic for a live building (you always know your own trailing readings) but means this is a short-horizon, own-history-dependent forecast, not a from-scratch prediction with zero context.
- `floor_id` itself gets very low direct importance (~0.1-0.3%). This does not mean floor identity doesn't matter -- it means floor-specific behavior is already captured indirectly through each floor's own lag/rolling values (a floor's lag_168h is computed from that floor's own history), so the explicit `floor_id` split adds little on top. Step 12 digs into floor-specific patterns more directly.
