# SHAP Explainability

SHAP (`TreeExplainer`) applied to the trained RandomForest model on a stratified sample of the 2019 test set (600 rows per floor). This is the model's XAI deliverable: it shows not just *that* the model predicts a given value, but *why*, in terms a building operator can act on.

## Global feature importance (SHAP)

|                 |      0 |
|:----------------|-------:|
| lag_168h        | 18.164 |
| lag_1h          | 17.62  |
| lag_24h         |  2.402 |
| hour            |  1.928 |
| roll_mean_3h    |  1.08  |
| temperature     |  0.923 |
| lux             |  0.509 |
| floor_id        |  0.304 |
| day_of_week     |  0.296 |
| roll_mean_24h   |  0.28  |
| is_weekend      |  0.104 |
| humidity        |  0.077 |
| month           |  0.068 |
| has_sensor_data |  0.014 |

Broadly agrees with the RandomForest's built-in importances from step 9/12 (`lag_168h` and `lag_1h` dominate), which is a good consistency check -- two different importance methods agree.

Figures: `14_shap_global_importance.png` (bar), `15_shap_summary_beeswarm.png` (beeswarm -- shows direction of effect, not just magnitude).

## Hour-of-day: Floor 1 vs. the rest

- Floors 2-7: the `hour` feature's SHAP contribution is noisy on its own (it's a minor feature, mean |SHAP| far below the lag features), but its **peak positive contribution is at 7:00**, in the same ramp-up window flagged by the raw consumption profile below and by step 12's peak-hour figures -- a useful cross-check even though `hour` alone doesn't cleanly separate day from night the way the lag features already do.
- Floor 1's hour-of-day SHAP contribution peaks at **18:00** instead, and its curve shape (figure `17_shap_hour_floor1_vs_rest.png`) does not resemble the other floors' -- the model itself has learned Floor 1 runs on a different schedule, not just that its scale differs.

## Temperature

- Correlation between `temperature` and its SHAP contribution (rows with real sensor data only): -0.38. A positive correlation would support the naive 'AC load rises with ambient temperature' story, but that is NOT what this shows.

**This is a real, moderately negative relationship, and the likely explanation matters**: the `S1` sensors in this dataset measure *indoor room* temperature/humidity/lux, not outdoor weather (see `data_inspection.md`). A room that is being actively cooled reads a *lower* temperature precisely while its AC is working hardest (highest energy draw); a room with the AC off or idle drifts *warmer* while drawing little power. That produces exactly this pattern -- temperature looks like it's *suppressing* the prediction as it rises, but the actual causal arrow plausibly runs the other way (AC-on lowers both room temperature and, mechanically, is the thing that raised energy in the first place). This is a reminder that SHAP explains the model's learned association, not necessarily the physical causal direction, and that indoor sensors answer a different question than outdoor weather would.

Figures: `16_shap_dependence_<feature>.png` for the top 3 features by importance.

## Energy-saving recommendations (SHAP-backed)

1. **Target load-shifting/demand-response at floors 2-7 during hours 8:00-16:00** -- this is the above-average-demand window in actual observed consumption for those floors, and the model's `hour` feature peaks positively right at its edge (7:00, the ramp-up hour just before it), a consistent cross-check even though the dominant model drivers are the lag features rather than hour-of-day directly. Shifting flexible/non-essential loads (e.g. batch equipment, non-urgent charging) outside this window, or pre-cooling before it starts (ahead of 7:00) rather than reactively cooling during it, targets the building's single largest, most consistently identified lever.
2. **Investigate Floor 1's evening-driven load specifically** -- its SHAP profile is not explained by the office-hours pattern that drives every other floor. Since it also has no working temperature/humidity/lux sensors in this dataset (`has_sensor_data`=0), an operator cannot currently correlate its 18:00 peak with ambient conditions -- installing sensors on Floor 1 would let both future modeling and manual investigation confirm whether this is legitimate equipment load or an opportunity (e.g. equipment left running after hours).
3. **Don't use this dataset's indoor temperature sensor to justify setpoint changes directly** -- its SHAP relationship with energy is negative, most plausibly because it measures room temperature *after* cooling (AC running hard -> room reads cooler -> high energy at the same time), not outdoor conditions driving demand. To get an actionable 'raise the setpoint when it's hot outside' recommendation, the building would need an *outdoor* weather feed -- worth adding as a future data source rather than over-interpreting the indoor sensor that exists today.
4. **Don't over-invest in weekend/off-hours efficiency measures** -- both SHAP and the raw EDA agree consumption outside the identified peak window is already low for floors 2-7; the biggest, most model-confirmed opportunity is squarely inside the office-hours peak, not the overnight baseline.
