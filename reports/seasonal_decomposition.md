# Seasonal Trend Decomposition

STL decomposition of `data/processed/cleaned_hourly.csv` (post-cleaning). Two resolutions, matching the two cycles EDA found: hourly/period=24 (daily occupancy cycle, per floor) and daily/period=7 (weekly cycle, combined building total).

## Daily cycle (hourly STL, period=24h)

`seasonal_share` / `residual_share` = fraction of that floor's variance explained by the recurring daily pattern vs. leftover noise -- higher seasonal_share means the floor's load is more predictably occupancy-driven.

|   floor_id |   seasonal_share |   residual_share |   trend_min |   trend_max |
|-----------:|-----------------:|-----------------:|------------:|------------:|
|          1 |            0.679 |            0.19  |        60.9 |       179.5 |
|          2 |            0.997 |            0.46  |         0.7 |        46.2 |
|          3 |            0.882 |            0.474 |        -0.1 |        25.8 |
|          4 |            0.856 |            0.419 |        -1.4 |        26.5 |
|          5 |            0.761 |            0.404 |         0.7 |        21.8 |
|          6 |            0.995 |            0.46  |         0.5 |        37.2 |
|          7 |            1.032 |            0.422 |        -8.9 |        59.4 |

Figures: `07_stl_hourly_floor3.png` (worked example), `08_stl_trend_all_floors.png` (deseasonalized trend level, all floors overlaid -- this is the cleanest view of each floor's *underlying* load level with the daily cycle removed).

## Weekly cycle (daily STL, period=7d, combined total)

- Seasonal (weekly) share of variance: 0.742
- Residual share of variance: 0.249
- Trend component peaks around **2018-08-08** and troughs around **2019-02-01**.
- This trend component is the deseasonalized (day-of-week removed) view of the monthly pattern already visible in `05_monthly_seasonal.png` -- consistent with a cool-season dip and hot-season rise driven by AC load, now isolated from the weekday/weekend swing.
- Two dips stand out beyond the ordinary seasonal wave: a sharp drop to ~3,900 around **late Jan-early Feb 2019**, plausibly a university low-occupancy/exam-break period rather than a data artifact (all 7 floors are present and contributing in this window). Worth flagging as a hypothesis, not a confirmed cause -- we have no occupancy calendar to verify it against.
- **Important caveat**: the trend also steps down and flattens from late October 2019 onward. This coincides exactly with Floor 6's data truncation (2019-10-22, see `data_inspection.md`) -- from that date the 'combined total' is a sum over only 6 floors instead of 7, which mechanically lowers it regardless of any real occupancy change. **This is a floor-count artifact, not a genuine seasonal decline** -- don't read the Nov-Dec 2019 section of this trend as 'the building used less energy'; it means 'one fewer floor is represented'.

Figure: `09_stl_daily_combined.png`.
