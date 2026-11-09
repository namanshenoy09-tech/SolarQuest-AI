import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pandas as pd

protocol = Path("reports/cyprus_may_protocol.md")
feasibility = Path("reports/cyprus_may_feasibility.json")
qc_path = Path("reports/cyprus_may_hourly_qc.csv")
downloader = Path("fetch_cyprus_may_weather.py")
out = Path("reports/cyprus_may_weather_join.json")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if out.exists():
    raise FileExistsError(
        f"{out} already exists. Preserve it before rerunning."
    )

committed = subprocess.check_output(
    ["git", "show", "HEAD:reports/cyprus_may_protocol.md"]
)

report = json.loads(feasibility.read_text(encoding="utf-8"))

if protocol.read_bytes() != committed or sha(protocol) != report["protocol_sha256"]:
    raise ValueError("Protocol differs from the committed feasibility check.")

for name, digest in report["input_sha256"].items():
    if sha(Path(name)) != digest:
        raise ValueError(f"Input changed since the feasibility check: {name}")

expected_downloader = (
    "9eeacc4941badb6e20b7781647f196b5"
    "6665843b6b6c3d8f1b7bd435dd9a8d83"
)

if sha(downloader) != expected_downloader:
    raise ValueError("Cyprus downloader changed; stop and share this error.")

# Reuse the verified downloader to validate the saved weather files.
config = runpy.run_path(str(downloader))

weather_times = []
paths = []
grids = []

for start, end in config["PERIODS"]:
    path = Path("data/cyprus") / f"weather_48h_{start}_to_{end}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    response = saved["response"]

    weather_times.extend(pd.to_datetime(
        response["hourly"]["time"], utc=True, errors="raise"
    ))

    paths.append(path)
    grids.append({
        "file": str(path),
        "latitude": response["latitude"],
        "longitude": response["longitude"]
    })

weather_times = pd.DatetimeIndex(weather_times)

expected = pd.date_range(
    "2025-05-17", "2025-06-01",
    freq="h", inclusive="left", tz="UTC"
)

if not weather_times.equals(expected):
    raise ValueError("Combined weather timestamps are missing or duplicated.")

qc = pd.read_csv(
    qc_path,
    usecols=["hour_start_local", "hour_end_utc", "status"]
)

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

calculated_utc = (
    qc["local_start"] + pd.Timedelta(hours=1)
).dt.tz_convert("UTC")

if not (qc["utc_end"] == calculated_utc).all():
    raise ValueError("QC hour ends do not match the assumed timezone.")

complete = qc[qc["status"] == "complete"].copy()

train = complete[
    complete["local_start"]
    < pd.Timestamp("2025-05-21 23:00", tz="Asia/Nicosia")
]

test = complete[
    (complete["local_start"] >= pd.Timestamp("2025-05-24", tz="Asia/Nicosia"))
    & (complete["local_start"] < pd.Timestamp("2025-06-01", tz="Asia/Nicosia"))
]

history = pd.DatetimeIndex(complete["local_start"])

both = (
    (test["local_start"] - pd.Timedelta(hours=72)).isin(history)
    & (test["local_start"] - pd.Timedelta(days=7)).isin(history)
)

train_match = train["utc_end"].isin(weather_times)
test_match = test["utc_end"].isin(weather_times)

result = {
    "warning": (
        "Weather timestamp coverage only. No model trained or scored. "
        "Timezone and meter alignment are provisional. The forecast grid "
        "is not necessarily the site; previous-day2 is not necessarily "
        "one simultaneously issued 48-hour forecast."
    ),
    "requested_site": {
        "latitude": 34.927423,
        "longitude": 33.475382
    },
    "response_grid_points": grids,
    "training_complete_hours": len(train),
    "training_weather_matched": int(train_match.sum()),
    "training_weather_unmatched": int((~train_match).sum()),
    "holdout_complete_hours": len(test),
    "holdout_weather_matched": int(test_match.sum()),
    "holdout_weather_unmatched": int((~test_match).sum()),
    "holdout_both_baselines_and_weather": int((both & test_match).sum()),
    "feasibility_floors_met": bool(
        int(train_match.sum()) >= 50
        and int((both & test_match).sum()) >= 20
    ),
    "protocol_sha256": sha(protocol),
    "feasibility_sha256": sha(feasibility),
    "join_script_sha256": sha(Path("cyprus_may_weather_join.py")),
    "downloader_sha256": sha(downloader),
    "weather_input_sha256": {str(path): sha(path) for path in paths}
}

with out.open("x", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
