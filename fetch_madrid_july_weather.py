import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FOLDER = Path("data/madrid")

FIELDS = [
    "temperature_2m_previous_day2",
    "shortwave_radiation_previous_day2"
]

PERIODS = [
    ("2025-07-22", "2025-07-28"),
    ("2025-07-29", "2025-07-30")
]

def check(saved, params):
    if saved["request_parameters"] != params:
        raise ValueError("Saved request parameters do not match.")

    payload = saved["response"]

    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("Expected UTC timestamps.")

    hourly = payload["hourly"]
    actual = pd.DatetimeIndex(
        pd.to_datetime(hourly["time"], utc=True, errors="raise")
    )

    end = pd.Timestamp(params["end_date"]) + pd.Timedelta(days=1)
    expected = pd.date_range(
        params["start_date"],
        end,
        freq="h",
        inclusive="left",
        tz="UTC"
    )

    if not actual.equals(expected):
        raise ValueError("Missing, duplicate or unexpected weather timestamps.")

    for key in FIELDS:
        values = hourly[key]
        valid = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        )

        if len(values) != len(expected) or not valid:
            raise ValueError(f"Missing or invalid values for {key}.")

    if any(value < 0 for value in hourly[FIELDS[1]]):
        raise ValueError("Negative forecast radiation.")

    return len(expected)

FOLDER.mkdir(parents=True, exist_ok=True)

for start, end in PERIODS:
    params = {
        "latitude": 40.4053889,
        "longitude": -3.7028056,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(FIELDS),
        "timezone": "UTC"
    }

    path = FOLDER / f"weather_48h_{start}_to_{end}.json"

    try:
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            count = check(saved, params)
            print(f"Already saved and checked: {path} ({count} hours)")
            continue

        print(f"Downloading {start} through {end}...", flush=True)
        response = requests.get(URL, params=params, timeout=(20, 60))
        response.raise_for_status()

        saved = {
            "source_url": URL,
            "request_parameters": params,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "response": response.json()
        }

        count = check(saved, params)

        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(saved), encoding="utf-8")
        temp.replace(path)

        print(f"Saved: {path} ({count} hours)", flush=True)

    except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
        print(f"Stopped at {start}: {exc}")
        print("Earlier completed files are safe. Share the error before retrying.")
        raise SystemExit(1)

print("July holdout weather chunks are ready.")

# Check timestamp coverage without evaluating model performance.
qc = pd.read_csv(
    "reports/madrid_july_hourly_qc.csv",
    usecols=["hour_start_local", "hour_end_utc", "status"]
)

qc["local_start"] = pd.to_datetime(
    qc["hour_start_local"], errors="raise"
)

holdout = qc[
    (qc["local_start"] >= "2025-07-23")
    & (qc["local_start"] < "2025-07-31")
    & (qc["status"] == "complete")
].copy()

holdout["utc_time"] = pd.to_datetime(
    holdout["hour_end_utc"], utc=True, errors="raise"
)

weather_times = []

for start, end in PERIODS:
    path = FOLDER / f"weather_48h_{start}_to_{end}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    weather_times.extend(saved["response"]["hourly"]["time"])

weather_index = pd.DatetimeIndex(
    pd.to_datetime(weather_times, utc=True, errors="raise")
)

if weather_index.has_duplicates:
    raise ValueError("Duplicate timestamps across July weather chunks.")

matched = holdout["utc_time"].isin(weather_index)

print(f"Complete July holdout hours: {len(holdout)}")
print(f"Weather matched: {int(matched.sum())}")
print(f"Weather unmatched: {int((~matched).sum())}")
