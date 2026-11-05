# Cyprus May 2025: exploratory site-specific evaluation protocol

This protocol is recorded after inspecting May's data-quality counts and one
training-period weather response, but before calculating forecast errors.
It is exploratory, not a pristine independent final validation.

## Data and provenance
- Source: https://doi.org/10.5281/zenodo.20800202, file 2025_05.csv.
- Site coordinates requested for weather: 34.927423, 33.475382.
  The weather provider can return a nearby grid point, not the site itself.
- Target: pv_power (W), hourly arithmetic mean of 120 unique, on-grid,
  finite 30-second readings only. Missing readings are not zero.
- Source offsets agree with Asia/Nicosia in the inspected May file. Using
  that timezone and joining weather to the local hour's UTC end are
  provisional assumptions; measurement and interval semantics remain unconfirmed.
- SHA-256 raw CSV: 8ceff70f2088d5ea1c72f5a5915355ff03981b2343153c49589a18ca380d6faa
- SHA-256 audit script: 2c6f7de5e37c76764ed0608abb5788d18c1e0ca687c9668c0e3825019c1f60db
- SHA-256 hourly QC CSV: a31f424ef1a4e590843dc6f21da21bcd5e3bd708775399c136e5ab70ae225a1a
- SHA-256 audit summary: c0c55e1216e3dd8a3c5662bfcb7378e47293d2b86d9cc9db6a07b640ee458832

## Fixed split and model
- Fit a new Cyprus-specific ordinary linear regression, not Madrid model transfer.
- Train on quality-approved hours with local start before 2025-05-21 23:00.
- Test May 24–31, local start >= 2025-05-24 00:00 and < 2025-06-01 00:00.
- Features: temperature_2m_previous_day2, shortwave_radiation_previous_day2,
  and sine/cosine of local start hour. Predict nonnegative power, forcing zero
  where forecast radiation is <= 0, as in the Madrid experiment.
- Use archived Open-Meteo previous-day2 weather at the hour-end UTC key.
  These fields are not necessarily a single simultaneously issued 48-hour forecast.

## Comparison and safeguards
- Compare model, 72-hour persistence, and seven-day persistence on the same
  complete target hours with weather and both complete baseline readings.
- Assume baseline telemetry is timely; count missing weather and baselines.
- Report total calendar hours, eligible and paired coverage, paired MAE in W,
  positive-forecast-radiation MAE, and descriptive daily scores.
- Require at least 50 complete matched training rows and 20 paired test rows
  for the exploratory score. These are feasibility floors, not proof of reliability.
- Do not alter the split, quality rules, model or metric after seeing the scores.
  If the test is infeasible, report that rather than selecting a better window.
- Later work requires fresh untouched periods and additional sites. No claims of
  operational readiness, causation, financial savings or cross-site transfer.
