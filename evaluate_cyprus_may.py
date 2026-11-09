"""Pre-specified exploratory Cyprus site-specific test; not model transfer."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

protocol = Path("reports/cyprus_may_protocol.md")
feasibility = Path("reports/cyprus_may_feasibility.json")
join_file = Path("reports/cyprus_may_weather_join.json")
qc_file = Path("reports/cyprus_may_hourly_qc.csv")
downloader = Path("fetch_cyprus_may_weather.py")
join_script = Path("cyprus_may_weather_join.py")
out = Path("reports/cyprus_may_holdout.json")

fields = [
    "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day2"
]

periods = [
    ("2025-05-17", "2025-05-23"),
    ("2025-05-24", "2025-05-30"),
    ("2025-05-31", "2025-05-31")
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if out.exists():
    raise FileExistsError(
        f"{out} already exists. Preserve it before rerunning."
    )

committed = subprocess.check_output(
    ["git", "show", "HEAD:reports/cyprus_may_protocol.md"]
)

feas = json.loads(feasibility.read_text(encoding="utf-8"))
joined_report = json.loads(join_file.read_text(encoding="utf-8"))

if (
    protocol.read_bytes() != committed
    or sha(protocol) != feas["protocol_sha256"]
    or sha(protocol) != joined_report["protocol_sha256"]
):
    raise ValueError("Protocol differs from the committed checks.")

if sha(feasibility) != joined_report["feasibility_sha256"]:
    raise ValueError("Feasibility report has changed.")

for name, digest in feas["input_sha256"].items():
    if sha(Path(name)) != digest:
        raise ValueError(f"Input changed since feasibility check: {name}")

if (
    sha(downloader) != joined_report["downloader_sha256"]
    or sha(join_script) != joined_report["join_script_sha256"]
):
    raise ValueError("Weather downloader or join script has changed.")

parts = []

for start, end in periods:
    path = Path("data/cyprus") / f"weather_48h_{start}_to_{end}.json"

    if sha(path) != joined_report["weather_input_sha256"].get(str(path)):
        raise ValueError(f"Weather file changed since the join check: {path}")

    saved = json.loads(path.read_text(encoding="utf-8"))

    params = {
        "latitude": 34.927423,
        "longitude": 33.475382,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(fields),
        "timezone": "UTC"
    }

    if (
        saved["source_url"] != "https://previous-runs-api.open-meteo.com/v1/forecast"
        or saved["request_parameters"] != params
    ):
        raise ValueError(f"Unexpected weather source or request: {path}")

    payload = saved["response"]

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError(f"Weather is not UTC: {path}")

    units = payload.get("hourly_units", {})

    if units.get(fields[0]) != "°C" or units.get(fields[1]) != "W/m²":
        raise ValueError(f"Unexpected weather units: {path}")

    hourly = payload["hourly"]

    parts.append(pd.DataFrame({
        "utc_end": pd.to_datetime(
            hourly["time"], utc=True, errors="raise"
        ),
        "temperature": hourly[fields[0]],
        "radiation": hourly[fields[1]]
    }))

weather = pd.concat(parts, ignore_index=True).sort_values("utc_end")

expected_utc = pd.date_range(
    "2025-05-17", "2025-06-01",
    freq="h", inclusive="left", tz="UTC"
)

if not pd.DatetimeIndex(weather["utc_end"]).equals(expected_utc):
    raise ValueError("Weather timestamps missing, duplicated or unexpected.")

for col in ["temperature", "radiation"]:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

if (
    not np.isfinite(weather[["temperature", "radiation"]].to_numpy()).all()
    or (weather["radiation"] < 0).any()
):
    raise ValueError("Invalid weather values.")

qc = pd.read_csv(qc_file)

qc["local_start"] = pd.to_datetime(
    qc["hour_start_local"], utc=True, errors="raise"
).dt.tz_convert("Asia/Nicosia")

qc["utc_end"] = pd.to_datetime(
    qc["hour_end_utc"], utc=True, errors="raise"
)

calendar = pd.date_range(
    "2025-05-01", "2025-06-01",
    freq="h", inclusive="left", tz="Asia/Nicosia"
)

if not pd.DatetimeIndex(qc["local_start"]).sort_values().equals(calendar):
    raise ValueError("QC hours are missing or duplicated.")

expected_end = (
    qc["local_start"] + pd.Timedelta(hours=1)
).dt.tz_convert("UTC")

if not qc["utc_end"].equals(expected_end):
    raise ValueError("QC hour ends do not match the assumed timezone.")

complete = qc[qc["status"] == "complete"].copy()
complete["power_w"] = pd.to_numeric(
    complete["sampled_mean_power_w"], errors="coerce"
)

if (
    not np.isfinite(complete["power_w"]).all()
    or complete["utc_end"].duplicated().any()
):
    raise ValueError("Invalid power or duplicate complete-hour timestamps.")

cutoff = pd.Timestamp("2025-05-21 23:00", tz="Asia/Nicosia")
start = pd.Timestamp("2025-05-24", tz="Asia/Nicosia")
end = pd.Timestamp("2025-06-01", tz="Asia/Nicosia")

train_hours = complete[complete["local_start"] < cutoff]

test_hours = complete[
    (complete["local_start"] >= start)
    & (complete["local_start"] < end)
]

def join_weather(frame):
    result = frame.merge(
        weather, on="utc_end", how="left", validate="one_to_one"
    ).sort_values("local_start")

    missing = result[["temperature", "radiation"]].isna().any(axis=1)
    result = result.loc[~missing].copy()

    hour = result["local_start"].dt.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return result, int(missing.sum())

train, train_weather_missing = join_weather(train_hours)
test, test_weather_missing = join_weather(test_hours)

history = complete.set_index("local_start")["power_w"]

for name, hours in [("baseline_72h", 72), ("baseline_7d", 168)]:
    prior = pd.DatetimeIndex(
        test["local_start"] - pd.Timedelta(hours=hours)
    )
    test[name] = history.reindex(prior).to_numpy()

paired = test.dropna(subset=["baseline_72h", "baseline_7d"]).copy()

if (
    len(train) != joined_report["training_weather_matched"]
    or len(test) != joined_report["holdout_weather_matched"]
    or len(paired) != joined_report["holdout_both_baselines_and_weather"]
):
    raise ValueError("Counts differ from the weather join report.")

if len(train) < 50 or len(paired) < 20:
    raise ValueError(
        f"Feasibility floors not met: {len(train)} train, "
        f"{len(paired)} paired. Do not change the split."
    )

features = ["temperature", "radiation", "hour_sin", "hour_cos"]

model = LinearRegression().fit(train[features], train["power_w"])

test["model"] = np.where(
    test["radiation"] <= 0,
    0,
    np.maximum(0, model.predict(test[features]))
)

paired = test.dropna(subset=["baseline_72h", "baseline_7d"]).copy()

def score(frame):
    if frame.empty:
        return None

    result = {"rows": len(frame)}

    for name in ["model", "baseline_72h", "baseline_7d"]:
        result[name + "_mae_w"] = round(
            float(mean_absolute_error(frame["power_w"], frame[name])), 3
        )

    return result

daily = {}

for day in pd.date_range(
    "2025-05-24", "2025-06-01",
    freq="D", inclusive="left", tz="Asia/Nicosia"
):
    date = day.date()

    daily[str(date)] = {
        "calendar_hours": 24,
        "complete_target_hours": int(
            (test_hours["local_start"].dt.date == date).sum()
        ),
        "weather_matched_hours": int(
            (test["local_start"].dt.date == date).sum()
        ),
        "paired_scores": score(
            paired[paired["local_start"].dt.date == date]
        )
    }

inputs = [
    protocol,
    feasibility,
    join_file,
    downloader,
    join_script,
    qc_file,
    Path("cyprus_may_feasibility.py"),
    Path("evaluate_cyprus_may.py")
]

inputs += [Path(name) for name in joined_report["weather_input_sha256"]]

result = {
    "warning": (
        "Exploratory same-site retraining, not cross-site transfer or "
        "operational validation. Only selected complete hours are scored; "
        "missing is not zero. Meter and weather alignment are provisional; "
        "previous-day2 is not necessarily one simultaneously issued "
        "48-hour forecast. Baselines assume timely telemetry. "
        "Training spans only a few days."
    ),
    "python_version": sys.version.split()[0],
    "sklearn_version": sklearn.__version__,
    "training_hour_start_exclusive_cutoff_local": (
        "2025-05-21 23:00:00 Asia/Nicosia"
    ),
    "training_complete_hours": len(train_hours),
    "training_weather_unmatched": train_weather_missing,
    "training_rows": len(train),
    "training_local_days": int(train["local_start"].dt.date.nunique()),
    "holdout_start_local": "2025-05-24 00:00:00 Asia/Nicosia",
    "holdout_end_exclusive_local": "2025-06-01 00:00:00 Asia/Nicosia",
    "total_calendar_hours": 192,
    "complete_target_hours": len(test_hours),
    "holdout_weather_unmatched": test_weather_missing,
    "holdout_weather_matched": len(test),
    "missing_72h_baseline": int(test["baseline_72h"].isna().sum()),
    "missing_7d_baseline": int(test["baseline_7d"].isna().sum()),
    "paired_all": score(paired),
    "paired_positive_forecast_radiation": score(
        paired[paired["radiation"] > 0]
    ),
    "daily_paired": daily,
    "input_sha256": {str(path): sha(path) for path in inputs}
}

with out.open("x", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
