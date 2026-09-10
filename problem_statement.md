# Problem Statement — Energy Consumption Prediction for Smart Buildings

## What we are forecasting

Hourly electrical energy consumption for each floor of a multi-floor
office building (CU-BEMS dataset: 7 floors, 2018-2019), where per-hour
energy for a floor is the summed power draw (kWh-equivalent from kW
readings) across all metered zones and end-uses (lighting, plug loads,
and air-conditioning) on that floor.

The model predicts `energy` for a given `(timestamp, floor_id)` pair,
using time-based context (hour of day, day of week, month, recent
history) and, where available, environmental readings (temperature,
humidity, illuminance) as inputs.

## Granularity

- **Temporal**: hourly. Raw sensor data is captured at near-minute
  frequency, which is too noisy and voluminous to model directly; hourly
  aggregation smooths sensor jitter while remaining fine-grained enough
  to capture daily occupancy cycles (working hours vs. overnight/weekend).
- **Spatial**: per floor. All zones/devices on a floor are combined into
  a single floor-level energy figure; floor identity is retained as a
  categorical feature so one model can learn floor-specific baseline load
  and behavior rather than training seven separate models.

## Why one combined model, not per-floor models

Floors share the same underlying drivers of consumption (time of day,
day of week, occupancy patterns, ambient conditions), but differ in
scale and equipment mix (e.g. number of AC units, zone count). Training
a single model across all floors with `floor_id` as a categorical
feature lets the model share statistical strength across floors (more
training data per pattern) while still learning floor-specific offsets
and interactions — closer to how a real building-management system would
be deployed and maintained (one model, not seven to keep in sync).

## Train/test protocol

Training uses all of 2018; testing/validation uses all of 2019. This is
a chronological, full-year-ahead split by design: it measures whether
the model generalizes across a full annual cycle (different occupancy
patterns, possible equipment changes, seasonal drift) rather than
memorizing interpolated patterns from a random split of the same year.
We expect this to be a harder, more realistic test than a random split,
and report results accordingly rather than optimizing the split for a
better-looking score.

## What "optimization insight" means as an output

The deliverable is not just a number (predicted kWh) — it is a
**plain-language, actionable explanation** of *why* the model predicts
what it does for a selected floor and time window, and what a building
operator could change as a result. Concretely, for a given
floor + date range, the app surfaces:

1. **The forecast** — predicted hourly energy consumption over the
   selected window.
2. **The explanation** — SHAP-based feature attributions showing which
   inputs (e.g. hour of day, day of week, temperature, recent load) push
   the prediction up or down for that specific forecast, satisfying the
   project's explainable-AI (XAI) requirement.
3. **The recommendation** — a translation of the top SHAP drivers into
   concrete, human-readable suggestions (e.g. "AC load dominates midday
   predictions on this floor — consider raising setpoint by 1-2°C during
   13:00-16:00" or "weekend baseline load is disproportionately high
   relative to weekday occupancy — check for equipment left running").

This turns a black-box regression into a decision-support tool: a
building manager should be able to look at one floor/date-range view and
understand both *what* consumption is expected and *what lever* would
most plausibly reduce it.
