import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

# Recreates the same June-trained model.
# Importing also refreshes the earlier July report.
import evaluate_july as june

root = Path("data")

paths = [
    root / "weather_weekly/2025-07-15_to_2025-07-21.json",
    root / "weather_holdout/week1.json",
    root / "weather_holdout/week2.json"
]

# Load and check the archived weather data.
parts = []

for path in paths:
    with path.open() as f:
        obj = json.load(f)

    # The weekly file is wrapped; the two new downloads are raw.
    payload = obj.get("response", obj)

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError(f"Expected UTC weather timestamps in {path}.")

    hourly = payload["hourly"]

    parts.append(pd.DataFrame({
        "utc_time": pd.to_datetime(
            hourly["time"], utc=True, errors="raise"
        ),
        "temperature": hourly["temperature_2m_previous_day2"],
        "radiation": hourly["shortwave_radiation_previous_day2"]
    }))

weather = pd.concat(parts, ignore_index=True).sort_values("utc_time")

expected_weather = pd.date_range(
    "2025-07-15", "2025-08-01",
    freq="h", inclusive="left", tz="UTC"
)

if not pd.DatetimeIndex(weather["utc_time"]).equals(expected_weather):
    raise ValueError("Weather coverage has missing or duplicate hours.")

for col in ["temperature", "radiation"]:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

if not np.isfinite(weather[["temperature", "radiation"]].to_numpy()).all():
    raise ValueError("Invalid or missing weather values.")

if (weather["radiation"] < 0).any():
    raise ValueError("Negative forecast radiation found.")

# Load July generation, including history needed by the baselines.
generation = pd.read_csv(root / "rooftop_generation.csv")
generation["local_time"] = pd.to_datetime(
    generation["Data"], errors="raise"
)
generation["Generation"] = pd.to_numeric(
    generation["Generation"], errors="coerce"
)

generation = generation[
    (generation["local_time"] >= "2025-07-01")
    & (generation["local_time"] < "2025-08-01")
].copy().sort_values("local_time")

expected_generation = pd.date_range(
    "2025-07-01", "2025-08-01", freq="h", inclusive="left"
)

if not pd.DatetimeIndex(generation["local_time"]).equals(expected_generation):
    raise ValueError("July generation has missing or duplicate timestamps.")

if not np.isfinite(generation["Generation"]).all():
    raise ValueError("Missing or invalid generation values.")

if (generation["Generation"] < 0).any():
    raise ValueError("Negative generation values found.")

# Working assumption only: the spreadsheet uses Kyiv local time.
generation["utc_time"] = (
    generation["local_time"]
    .dt.tz_localize("Europe/Kyiv")
    .dt.tz_convert("UTC")
)

target = generation[
    generation["local_time"] >= "2025-07-22"
].copy()

test = target.merge(
    weather, on="utc_time", how="left", validate="one_to_one"
)

if len(test) != 240 or test[["temperature", "radiation"]].isna().any().any():
    raise ValueError("Could not match weather for all 240 holdout hours.")

# Apply the same features and zero-radiation rule as before.
hour = test["local_time"].dt.hour
test["hour_sin"] = np.sin(2 * np.pi * hour / 24)
test["hour_cos"] = np.cos(2 * np.pi * hour / 24)

test["prediction"] = np.where(
    test["radiation"] <= 0,
    0,
    np.maximum(0, june.model.predict(test[june.features]))
)

# Compare with generation from 72 hours and seven days earlier.
history = generation.set_index("local_time")["Generation"]

for label, lag in [("baseline_72h", 72), ("baseline_7d", 168)]:
    prior_times = pd.DatetimeIndex(
        test["local_time"] - pd.Timedelta(hours=lag)
    )
    test[label] = history.reindex(prior_times).to_numpy()

if test[["baseline_72h", "baseline_7d"]].isna().any().any():
    raise ValueError("Some baseline readings are missing.")

def scores(df):
    if df.empty:
        return None

    return {
        "rows": len(df),
        "model_mae": round(float(mean_absolute_error(
            df["Generation"], df["prediction"]
        )), 3),
        "baseline_72h_mae": round(float(mean_absolute_error(
            df["Generation"], df["baseline_72h"]
        )), 3),
        "baseline_7d_mae": round(float(mean_absolute_error(
            df["Generation"], df["baseline_7d"]
        )), 3)
    }

inputs = (
    [root / "rooftop_generation.csv",
     root / "weather_48h_june_2025.json"]
    + june.weeks
    + paths[1:]
    + [Path("evaluate_july.py"), Path("evaluate_late_july.py")]
)

report = {
    "warning": (
        "Exploratory same-site test. Time zone, source units and "
        "hourly interval conventions are unconfirmed. Baselines "
        "assume timely telemetry. Not an operational validation."
    ),
    "holdout_local_start": str(test["local_time"].min()),
    "holdout_local_end": str(test["local_time"].max()),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "training_rows": len(june.train),
    "last_training_local_time": str(june.train["local_time"].max()),
    "all_hours": scores(test),
    "positive_forecast_radiation": scores(test[test["radiation"] > 0]),
    "input_sha256": {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
}

Path("reports").mkdir(exist_ok=True)
output = Path("reports/late_july_holdout.json")
output.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n--- LATE JULY HOLDOUT: ERRORS IN SOURCE UNITS ---")
print(json.dumps(report, indent=2))
print(f"Saved: {output}")
