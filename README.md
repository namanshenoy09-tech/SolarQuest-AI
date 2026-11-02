# SolarQuest-AI — research prototype

SolarQuest explores solar generation prediction and historical plant underperformance. **This is an experimental prototype, not a production operational, trading or fault-diagnosis system.**

## What works in this version

- Manual model inputs for an illustrative training-plant prediction.
- An Open-Meteo weather experiment for the next 48 full UTC hours. It assumes the model's irradiation units match shortwave radiation divided by 1,000, estimates module temperature, and is not a validated 48-hour forecast. Enter the actual training plant's coordinates and local time zone; the defaults are placeholders.
- CSV upload for comparing independently supplied hourly actual and expected values. This is not a live SCADA connection, and deviations do not identify their causes. Shortfall values are illustrative, not proven lost revenue or savings.
- An expanding-window chronological benchmark of **historical measured sensor inputs**, with a previous-day baseline, MAE, RMSE, R² and optional capacity-normalized MAE. It does not evaluate 48-hour-ahead forecasts.

See [APPLY_CHANGES.md](APPLY_CHANGES.md) for installation, training, tests, CSV requirements, limitations and next steps.

## Important limitations

No SCADA integration, verified alerting, field validation, multi-site generalization, market settlement, equipment fault classification, or 48-hour forecast backtest is implemented here. There is no capacity scaling, ROI claim or CO₂ estimate. Model and source data units, site capacity, licensing, suitability, missing-data completeness and weather-feature compatibility must be verified before use. The notebook's old workflows and output claims are historical, not validation of this release.

Only load model files from a source you trust: `joblib` / pickle files can execute code when loaded. The application never accepts model uploads.

License: MIT (see LICENSE).
