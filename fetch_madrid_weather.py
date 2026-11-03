import json
import math
import time
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
    ("2025-05-08", "2025-05-14"),
    ("2025-05-15", "2025-05-21"),
    ("2025-05-22", "2025-05-28"),
    ("2025-05-29", "2025-05-31")
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
        params["start_date"], end,
        freq="h", inclusive="left", tz="UTC"
    )

    if not actual.equals(expected):
        raise ValueError("Missing, duplicate or unexpected timestamps.")

    for key in FIELDS:
        values = hourly[key]
        if len(values) != len(expected) or not all(
            isinstance(v, (int, float)) and math.isfinite(v)
            for v in values
        ):
            raise ValueError(f"Missing or invalid values for {key}.")

    if any(v < 0 for v in hourly[FIELDS[1]]):
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

        response = requests.get(URL, params=params, timeout=(20, 45))
        response.raise_for_status()

        saved = {
            "source_url": URL,
            "request_parameters": params,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "response": response.json()
        }

        count = check(saved, params)

        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(saved), encoding="utf-8")
        temporary.replace(path)

        print(f"Saved: {path} ({count} hours)", flush=True)
        time.sleep(1)

    except (
        requests.RequestException, ValueError, KeyError, TypeError, OSError
    ) as exc:
        print(f"Stopped at {start}: {exc}")
        print("Earlier completed files are safe.")
        raise SystemExit(1)

print("All remaining May chunks are ready!")
