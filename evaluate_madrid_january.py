"""Fixed May-trained Madrid model evaluated on January 23–30."""
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
may_qc = Path("reports/madrid_may_hourly_qc.csv")
january_qc = Path("reports/madrid_january_hourly_qc.csv")
may_report = Path("reports/madrid_holdout_202505.json")
out = Path("reports/madrid_holdout_202601.json")

if out.exists():
    raise FileExistsError(
        f"{out} already exists. Preserve it before rerunning."
    )

may_files = [
    root / "weather_48h_20250430_20250507.json"
] + [
    root / f"weather_48h_{start}_to_{end}.json"
    for start, end in [
        ("2025-05-08", "2025-05-14"),
        ("2025-05-15", "2025-05-21"),
        ("2025-05-22", "2025-05-28"),
        ("2025-05-29", "2025-05-31")
    ]
]

january_files = [
    root / f"weather_48h_{start}_to_{end}.json"
    for start, end in [
        ("2026-01-22", "2026-01-28"),
        ("2026-01-29", "2026-01-30")
    ]
]

fields = [
    "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day2"
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

previous = json.loads(may_report.read_text(encoding="utf-8"))

# Confirm the saved May inputs have not changed.
for name, expected in previous["input_sha256"].items():
    if sha(Path(name)) != expected:
        raise ValueError(
            f"May input changed since the original report: {name}"
        )

if (
    previous["train_rows"] != 314
    or previous["model_mae_all_eligible_w"] != 54.841
):
    raise ValueError("Unexpected saved May result. Stop and investigate.")

def load_weather(files, start, end):
    parts = []

    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        params = saved["request_parameters"]

        if (
            abs(float(params["latitude"]) - 40.4053889) > 1e-6
            or abs(float(params["longitude"]) + 3.7028056) > 1e-6
            or params["timezone"] != "UTC"
            or params["hourly"] != ",".join(fields)
        ):
            raise ValueError(f"Unexpected weather request in {path}.")

        payload = saved["response"]

        if payload.get("utc_offset_seconds") != 0:
            raise ValueError(f"Weather is not UTC in {path}.")

        hourly = payload["hourly"]

        parts.append(pd.DataFrame({
            "utc_time": pd.to_datetime(
                hourly["time"], utc=True, errors="raise"
            ),
            "temperature": hourly[fields[0]],
            "radiation": hourly[fields[1]]
        }))

    result = pd.concat(parts, ignore_index=True).sort_values("utc_time")
    expected = pd.date_range(
        start, end, freq="h", inclusive="left", tz="UTC"
    )

    if not pd.DatetimeIndex(result["utc_time"]).equals(expected):
        raise ValueError(
            "Weather timestamps are missing, duplicated or unexpected."
        )

    for col in ["temperature", "radiation"]:
        result[col] = pd.to_numeric(result[col], errors="coerce")

    if (
        not np.isfinite(result[["temperature", "radiation"]].to_numpy()).all()
        or (result["radiation"] < 0).any()
    ):
        raise ValueError("Invalid weather values.")

    return result

def load_qc(path, start, end):
    qc = pd.read_csv(path)
    qc["local_start"] = pd.to_datetime(
        qc["hour_start_local"], errors="raise"
    )

    expected = pd.date_range(start, end, freq="h", inclusive="left")

    if not pd.DatetimeIndex(qc["local_start"]).sort_values().equals(expected):
        raise ValueError(f"Missing or duplicate QC hours in {path}.")

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

    if (
        not qc["utc_time"].equals(calculated_utc)
        or qc["utc_time"].duplicated().any()
        or not np.isfinite(qc["power_w"]).all()
    ):
        raise ValueError(f"Invalid complete-hour data in {path}.")

    return qc

def joined(qc, weather):
    result = qc.merge(
        weather, on="utc_time", how="left", validate="one_to_one"
    ).sort_values("local_start")

    if result[["temperature", "radiation"]].isna().any().any():
        raise ValueError("Weather did not match every complete target hour.")

    hour = result["local_start"].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return result

def predict(model, frame):
    return np.where(
        frame["radiation"] <= 0,
        0,
        np.maximum(0, model.predict(frame[features]))
    )

def score(frame):
    if frame.empty:
        return None

    result = {"rows": len(frame)}

    for name in ["model", "baseline_72h", "baseline_7d"]:
        result[name + "_mae_w"] = round(
            float(mean_absolute_error(frame["power_w"], frame[name])), 3
        )

    return result

def add_baselines(frame, history):
    frame = frame.copy()
    history = history.set_index("local_start")["power_w"]

    for name, hours in [("baseline_72h", 72), ("baseline_7d", 168)]:
        prior = pd.DatetimeIndex(
            frame["local_start"] - pd.Timedelta(hours=hours)
        )
        frame[name] = history.reindex(prior).to_numpy()

    return frame

features = ["temperature", "radiation", "hour_sin", "hour_cos"]

may = load_qc(may_qc, "2025-05-01", "2025-06-01")
may_weather = load_weather(may_files, "2025-04-30", "2025-06-01")
may_data = joined(may, may_weather)

train = may_data[
    may_data["local_start"] < pd.Timestamp("2025-05-21 23:00:00")
]

if (
    len(train) != 314
    or str(train["local_start"].max()) != "2025-05-21 21:00:00"
):
    raise ValueError("Training rows do not match the original May experiment.")

model = LinearRegression().fit(train[features], train["power_w"])

# Verify the refitted model reproduces the observed May comparison.
old_test = may_data[may_data["local_start"] >= "2025-05-24"].copy()
old_test["model"] = predict(model, old_test)

old_paired = add_baselines(old_test, may).dropna(
    subset=["baseline_72h", "baseline_7d"]
)

old_scores = score(old_paired)
old_reference = previous["paired_all"]

model_key = (
    "model_mae_w"
    if "model_mae_w" in old_reference
    else "prediction_mae_w"
)

for name, reference in [
    ("model_mae_w", model_key),
    ("baseline_72h_mae_w", "baseline_72h_mae_w"),
    ("baseline_7d_mae_w", "baseline_7d_mae_w"),
    ("rows", "rows")
]:
    if old_scores[name] != old_reference[reference]:
        raise ValueError(f"May reproduction mismatch: {name}.")

# January is test-only: the training cutoff and model are unchanged.
january = load_qc(january_qc, "2026-01-01", "2026-02-01")
january_weather = load_weather(january_files, "2026-01-22", "2026-01-31")

targets = january[
    (january["local_start"] >= "2026-01-23")
    & (january["local_start"] < "2026-01-31")
]

test = joined(targets, january_weather)
test["model"] = predict(model, test)
test = add_baselines(test, january)

paired = test.dropna(subset=["baseline_72h", "baseline_7d"])

if paired.empty:
    raise ValueError("No paired baseline rows.")

inputs = [
    may_report,
    january_qc,
    Path("madrid_january_qc.py"),
    Path("evaluate_madrid_january.py"),
    Path("fetch_madrid_january_weather.py"),
    Path("reports/madrid_future_test_protocol.md"),
    Path("reports/madrid_january_qc_summary.json")
] + january_files

result = {
    "warning": (
        "Exploratory same-site test. Selected complete hours only. "
        "Timing alignment is provisional. Previous-day2 weather fields "
        "are not necessarily one simultaneously issued 48-hour forecast. "
        "Baselines assume timely telemetry. Not operational validation "
        "or cross-site transfer."
    ),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "training_rows": len(train),
    "last_training_hour_start_local": str(train["local_start"].max()),
    "may_paired_reproduction": old_scores,
    "holdout_start_local": "2026-01-23 00:00:00",
    "holdout_end_exclusive_local": "2026-01-31 00:00:00",
    "total_calendar_hours": 192,
    "complete_target_hours": len(test),
    "model_mae_all_complete_w": round(
        float(mean_absolute_error(test["power_w"], test["model"])), 3
    ),
    "missing_72h_baseline_rows": int(test["baseline_72h"].isna().sum()),
    "missing_7d_baseline_rows": int(test["baseline_7d"].isna().sum()),
    "paired_all": score(paired),
    "paired_positive_forecast_radiation": score(
        paired[paired["radiation"] > 0]
    ),
    "daily_paired": {
        str(day.date()): {
            "calendar_hours": 24,
            "complete_target_hours": int(
                (test["local_start"].dt.date == day.date()).sum()
            ),
            "paired_scores": score(
                paired[paired["local_start"].dt.date == day.date()]
            )
        }
        for day in pd.date_range(
            "2026-01-23", "2026-01-31", freq="D", inclusive="left"
        )
    },
    "input_sha256": {str(path): sha(path) for path in inputs}
}

with out.open("x", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
