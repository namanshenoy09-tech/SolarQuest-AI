import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

root = Path("data")

# Load generation data.
g = pd.read_csv(root / "rooftop_generation.csv")
g["local_time"] = pd.to_datetime(g["Data"], errors="raise")
g["Generation"] = pd.to_numeric(g["Generation"], errors="raise")

# Select June and the first three weeks of July.
g = g[
    (g["local_time"] >= "2025-06-01")
    & (g["local_time"] < "2025-07-22")
].copy()

if g["local_time"].duplicated().any():
    raise ValueError("Duplicate generation timestamps.")

if not np.isfinite(g["Generation"]).all() or (g["Generation"] < 0).any():
    raise ValueError("Invalid generation values.")

# Working assumption, not yet confirmed by the dataset authors.
g["utc_time"] = (
    g["local_time"]
    .dt.tz_localize("Europe/Kyiv")
    .dt.tz_convert("UTC")
)

def read_weather(paths):
    parts = []

    for path in paths:
        with path.open() as f:
            hourly = json.load(f)["response"]["hourly"]

        parts.append(pd.DataFrame({
            "utc_time": pd.to_datetime(hourly["time"], utc=True),
            "temperature": hourly["temperature_2m_previous_day2"],
            "radiation": hourly["shortwave_radiation_previous_day2"]
        }))

    weather = pd.concat(parts, ignore_index=True)

    for col in ["temperature", "radiation"]:
        weather[col] = pd.to_numeric(weather[col], errors="coerce")

    if weather["utc_time"].isna().any() or weather["utc_time"].duplicated().any():
        raise ValueError("Invalid or duplicate weather timestamps.")

    if not np.isfinite(weather[["temperature", "radiation"]].to_numpy()).all():
        raise ValueError("Missing or invalid weather values.")

    if (weather["radiation"] < 0).any():
        raise ValueError("Negative forecast radiation found.")

    return weather

def join_weather(generation, weather):
    df = generation.merge(
        weather, on="utc_time", how="inner", validate="one_to_one"
    )

    hour = df["local_time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return df.sort_values("utc_time")

june = g[
    (g["local_time"] >= "2025-06-01")
    & (g["local_time"] < "2025-07-01")
]

july = g[
    (g["local_time"] >= "2025-07-01")
    & (g["local_time"] < "2025-07-22")
]

if len(june) != 720 or len(july) != 504:
    raise ValueError("Unexpected generation row count. Check the data.")

weeks = [
    root / "weather_weekly/2025-07-01_to_2025-07-07.json",
    root / "weather_weekly/2025-07-08_to_2025-07-14.json",
    root / "weather_weekly/2025-07-15_to_2025-07-21.json"
]

june_data = join_weather(
    june, read_weather([root / "weather_48h_june_2025.json"])
)
july_data = join_weather(july, read_weather(weeks))

# Recreate the same training cutoff used in the June experiment.
cutoff = pd.Timestamp("2025-06-21 00:00:00") - pd.Timedelta(hours=49)
train = june_data[june_data["local_time"] < cutoff]

if len(train) != 428 or len(july_data) != 501:
    raise ValueError("Unexpected joined row counts. Check weather coverage.")

features = ["temperature", "radiation", "hour_sin", "hour_cos"]

model = LinearRegression()
model.fit(train[features], train["Generation"])

test = july_data.copy()
test["prediction"] = np.where(
    test["radiation"] <= 0,
    0,
    np.maximum(0, model.predict(test[features]))
)

# Compare against generation from 72 hours earlier.
history = g.set_index("local_time")["Generation"]
baseline_times = pd.DatetimeIndex(
    test["local_time"] - pd.Timedelta(hours=72)
)
test["baseline"] = history.reindex(baseline_times).to_numpy()

paired = test.dropna(subset=["baseline"])

if paired.empty:
    raise ValueError("No rows available for baseline comparison.")

def score(df):
    if df.empty:
        return None

    return {
        "rows": len(df),
        "model_mae_source_units": round(
            float(mean_absolute_error(df["Generation"], df["prediction"])), 3
        ),
        "baseline_mae_source_units": round(
            float(mean_absolute_error(df["Generation"], df["baseline"])), 3
        )
    }

report = {
    "warning": (
        "Exploratory same-site test. Time zone, interval convention "
        "and generation units are unconfirmed."
    ),
    "training_rows": len(train),
    "last_training_time": str(train["local_time"].max()),
    "july_generation_rows": len(july),
    "july_weather_matched": len(test),
    "unmatched_july_rows": len(july) - len(test),
    "baseline_missing_rows": len(test) - len(paired),
    "all_paired_rows": score(paired),
    "positive_forecast_radiation_rows": score(
        paired[paired["radiation"] > 0]
    ),
    "baseline": (
        "Generation from 72 hours earlier; assumes timely availability."
    )
}

Path("reports").mkdir(exist_ok=True)
Path("reports/july_evaluation.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)

print(json.dumps(report, indent=2))
print("\nSaved to reports/july_evaluation.json")
