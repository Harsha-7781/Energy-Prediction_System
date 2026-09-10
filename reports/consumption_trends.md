# Consumption Trends: Plain-Language, Model-Backed Summary

Based on the trained RandomForest model's global feature importances and descriptive statistics from `data/processed/cleaned_hourly.csv`. This is the aggregate, whole-model picture; step 13 (SHAP) explains individual predictions.

## What drives consumption, in order of importance to the model

1. **Recent history dominates** (69% importance on `lag_168h` alone): the single best predictor of a floor's energy use this hour is that same floor's usage at the same hour last week. Buildings run on a repeating weekly routine (same opening hours, same occupancy pattern Monday to Monday), so last week's Tuesday-9am reading is a strong stand-in for this week's.
2. **Very recent readings matter too** (`lag_1h`, 26% importance): the previous hour's reading also carries real signal, capturing short-term momentum (a floor that was ramping up AC load 30 minutes ago is still likely doing so now).
3. **Time-of-day and day-of-week matter, but mostly through the lag features rather than as separate importance** -- the model doesn't need to be told 'it's Tuesday at 9am' explicitly when the lag features already encode what happened at that exact slot historically. The raw `hour` feature still carries some independent weight, describing the diurnal ramp shape.
4. **Floor identity itself carries little *extra* weight beyond history** -- not because floors don't differ (they clearly do, see below), but because each floor's own lag/rolling features already encode its scale and behavior. `floor_id` would matter more if the model had to predict a *new* floor with no history.
5. **Weather sensors (temperature/humidity/lux) and calendar context (month, weekend flag) contribute only marginally** at the aggregate level once recent history is available -- their effect is real (AC load does track ambient temperature -- see the seasonal swing figures below) but small relative to the dominant weekly-routine signal.

## Floor-specific differences

- **Floor 1**: weekday/weekend ratio 1.0x (mean 106.2 weekday vs 106.4 weekend), daily peak at **19:00** (170.4), trough at **08:00** (65.73), seasonal swing 42.1% (highest in 2018-08, lowest in 2019-12).
- **Floor 2**: weekday/weekend ratio 3.75x (mean 35.1 weekday vs 9.3 weekend), daily peak at **08:00** (59.0), trough at **05:00** (6.59), seasonal swing 59.2% (highest in 2018-07, lowest in 2019-01).
- **Floor 3**: weekday/weekend ratio 9.85x (mean 19.9 weekday vs 2.0 weekend), daily peak at **10:00** (39.6), trough at **02:00** (0.70), seasonal swing 47.7% (highest in 2019-09, lowest in 2019-02).
- **Floor 4**: weekday/weekend ratio 6.98x (mean 20.6 weekday vs 3.0 weekend), daily peak at **11:00** (45.4), trough at **02:00** (1.11), seasonal swing 52.9% (highest in 2018-08, lowest in 2019-04).
- **Floor 5**: weekday/weekend ratio 7.89x (mean 16.7 weekday vs 2.1 weekend), daily peak at **11:00** (38.5), trough at **03:00** (0.88), seasonal swing 39.7% (highest in 2019-09, lowest in 2019-02).
- **Floor 6**: weekday/weekend ratio 9.29x (mean 29.1 weekday vs 3.1 weekend), daily peak at **13:00** (63.3), trough at **03:00** (0.88), seasonal swing 28.5% (highest in 2019-05, lowest in 2018-12).
- **Floor 7**: weekday/weekend ratio 5.42x (mean 37.4 weekday vs 6.9 weekend), daily peak at **13:00** (80.1), trough at **03:00** (0.59), seasonal swing 90.7% (highest in 2019-05, lowest in 2019-02).

**Floor 1 stands out from the rest of the building**: every other floor (2-7) shows a strong weekday/weekend split (weekday load roughly 4-9x the weekend) and a single midday-ish peak typical of office occupancy. Floor 1's weekday/weekend ratio is much closer to 1 and its peak lands in the evening, not the middle of the day -- consistent with the EDA finding (step 4) that Floor 1 runs different load types (e.g. a large near-continuous system) rather than standard daytime office equipment. Any optimization recommendation for Floor 1 (step 13) needs to account for this -- advice tuned to a 9-to-5 office pattern would not fit how this floor actually behaves.

## Seasonal (weather-driven) component

Per-floor seasonal swings (highest vs. lowest monthly mean) range from 28.5% to 90.7%, smaller than the daily/weekly swing but consistent with AC-driven load responding to Thailand's cooler-season (roughly Dec-Feb) dip and hotter-season rise -- matching the STL trend component from step 6. This is a real but secondary driver compared to the occupancy cycle.
