# Fixed Madrid multi-period test protocol

- Site and channel: ETSIDI-UPM Madrid, T1_S1_Pac.
- Model: the same linear regression, features and prediction rules as the
  May and June evaluations; train only on the same 314 May hours.
- Test windows, local time: July 23–30, 2025 and January 23–30, 2026.
  Each window contains 192 calendar hours.
- Do not tune the model or change the remaining window after seeing results.
- Apply the existing strict five-second hourly quality checks. Do not
  interpolate or replace missing readings with zero.
- Compare the model, 72-hour baseline and seven-day baseline on the same
  quality-approved hours with both baseline readings and weather available.
- Report calendar-hour coverage, missing data, paired MAE in watts, positive
  forecast-radiation results and descriptive daily results. Report insufficient
  data rather than choosing a different window after seeing results.
- Use the same provisional hour-end weather alignment and previous-day2
  forecast fields as before. Baselines assume timely telemetry.
- This is a same-site historical experiment, not operational validation
  or cross-site model transfer.
