import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

root = Path("data/madrid")
qc_file = Path("reports/madrid_may_hourly_qc.csv")

weather_files = [
    root / "weather_48h_20250430_20250507.json",
    root / "weather_48h_2025-05-08_to_2025-05-14.json",
    root / "weather_48h_2025-05-15_to_2025-05-21.json",
    root / "weather_48h_2025-05-22_to_2025-05-28.json",
    root / "weather_48h_2025-05-29_to_2025-05-31.json"
]

fields = [
    "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day2"
]

# Load and validate weather files.
parts = []

for path in weather_files:
    with path.open() as f:
        saved = json.load(f)

    params = saved["request_parameters"]

    if (
        abs(float(params["latitude"]) - 40.4053889) > 1e-6
        or abs(float(params["longitude"]) + 3.7028056) > 1e-6
        or params["timezone"] != "UTC"
        or params["hourly"] != ",".join(fields)
    ):
        raise ValueError(f"Unexpected weather parameters in {path}.")

    payload = saved["response"]

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError(f"Weather timestamps are not UTC in {path}.")

    hourly = payload["hourly"]

    parts.append(pd.DataFrame({
        "utc_time": pd.to_datetime(
            hourly["time"], utc=True, errors="raise"
        ),
        "temperature": hourly[fields[0]],
        "radiation": hourly[fields[1]]
    }))

weather = pd.concat(parts, ignore_index=True).sort_values("utc_time")

expected_weather = pd.date_range(
    "2025-04-30", "2025-06-01",
    freq="h", inclusive="left", tz="UTC"
)

if not pd.DatetimeIndex(weather["utc_time"]).equals(expected_weather):
    raise ValueError("Weather has missing or duplicate timestamps.")

for col in ["temperature", "radiation"]:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

if not np.isfinite(weather[["temperature", "radiation"]].to_numpy()).all():
    raise ValueError("Weather contains missing or invalid values.")

if (weather["radiation"] < 0).any():
    raise ValueError("Negative forecast radiation found.")

# Load the May quality report.
qc = pd.read_csv(qc_file)
qc["local_start"] = pd.to_datetime(
    qc["hour_start_local"], errors="raise"
)

expected_local = pd.date_range(
    "2025-05-01", "2025-06-01",
    freq="h", inclusive="left"
)

if not pd.DatetimeIndex(qc["local_start"]).sort_values().equals(expected_local):
    raise ValueError("QC report has missing or duplicate May hours.")

# Keep only complete, quality-approved hours.
qc = qc[qc["status"] == "complete"].copy()
qc["power_w"] = pd.to_numeric(
    qc["sampled_mean_power_w"], errors="coerce"
)
qc["utc_time"] = pd.to_datetime(
    qc["hour_end_utc"], utc=True, errors="raise"
)

calculated_utc = (
    (qc["local_start"] + pd.Timedelta(hours=1))
    .dt.tz_localize("Europe/Madrid")
    .dt.tz_convert("UTC")
)

if not qc["utc_time"].equals(calculated_utc):
    raise ValueError("Hour-end UTC values don't match Madrid time.")

if not np.isfinite(qc["power_w"]).all() or qc["utc_time"].duplicated().any():
    raise ValueError("Invalid power or duplicate complete-hour timestamps.")

data = qc.merge(
    weather, on="utc_time", how="left", validate="one_to_one"
).sort_values("local_start")

if data[["temperature", "radiation"]].isna().any().any():
    raise ValueError("Weather did not match all complete Madrid hours.")

hour = data["local_start"].dt.hour
data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

features = ["temperature", "radiation", "hour_sin", "hour_cos"]

# Fixed split: later May is held out, with a gap before it.
test_start = pd.Timestamp("2025-05-24 00:00:00")
cutoff = test_start - pd.Timedelta(hours=49)

train = data[data["local_start"] < cutoff]
test = data[data["local_start"] >= test_start].copy()

if len(train) < 50 or len(test) < 20:
    raise ValueError("Insufficient complete training or test observations.")

model = LinearRegression()
model.fit(train[features], train["power_w"])

test["prediction"] = np.where(
    test["radiation"] <= 0,
    0,
    np.maximum(0, model.predict(test[features]))
)

# Baselines use quality-approved readings from earlier hours.
history = qc.set_index("local_start")["power_w"]

for name, hours in [("baseline_72h", 72), ("baseline_7d", 168)]:
    prior_times = pd.DatetimeIndex(
        test["local_start"] - pd.Timedelta(hours=hours)
    )
    test[name] = history.reindex(prior_times).to_numpy()

# Compare all three methods on the same available rows.
paired = test.dropna(subset=["baseline_72h", "baseline_7d"])

if paired.empty:
    raise ValueError("No test rows have both baseline readings.")

def scores(frame):
    if frame.empty:
        return None

    return {
        "rows": len(frame),
        "model_mae_w": round(float(mean_absolute_error(
            frame["power_w"], frame["prediction"]
        )), 3),
        "baseline_72h_mae_w": round(float(mean_absolute_error(
            frame["power_w"], frame["baseline_72h"]
        )), 3),
        "baseline_7d_mae_w": round(float(mean_absolute_error(
            frame["power_w"], frame["baseline_7d"]
        )), 3)
    }

inputs = [
    qc_file,
    Path("madrid_may_qc.py"),
    Path("evaluate_madrid.py")
] + weather_files

report = {
    "warning": (
        "Exploratory same-site test for T1_S1_Pac. Weather-to-meter "
        "alignment is provisional. Power is a five-second sampled "
        "mean in watts. Only complete hours are scored; missing hours "
        "are not zero. Baselines assume timely telemetry. This is not "
        "operational validation or cross-site model transfer."
    ),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "qc_complete_hours_in_may": len(qc),
    "train_rows": len(train),
    "last_training_hour_start_local": str(train["local_start"].max()),
    "holdout_start_local": str(test_start),
    "holdout_end_exclusive_local": "2025-06-01 00:00:00",
    "eligible_holdout_rows": len(test),
    "model_mae_all_eligible_w": round(float(mean_absolute_error(
        test["power_w"], test["prediction"]
    )), 3),
    "missing_72h_baseline_rows": int(test["baseline_72h"].isna().sum()),
    "missing_7d_baseline_rows": int(test["baseline_7d"].isna().sum()),
    "paired_all": scores(paired),
    "paired_positive_forecast_radiation": scores(
        paired[paired["radiation"] > 0]
    ),
    "input_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
}

Path("reports").mkdir(exist_ok=True)
output = Path("reports/madrid_holdout_202505.json")
output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSaved: {output}")
