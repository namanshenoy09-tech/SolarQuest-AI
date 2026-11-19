import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pandas as pd

protocol = Path("reports/cyprus_july_training_comparison_protocol.md")
manifest_path = Path("reports/cyprus_july_qc_manifest.json")
downloader = Path("fetch_cyprus_march_july_weather.py")
out = Path("reports/cyprus_july_weather_join.json")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if out.exists():
    raise FileExistsError(
        f"{out} already exists. Preserve it before rerunning."
    )

committed = subprocess.check_output([
    "git", "show",
    "HEAD:reports/cyprus_july_training_comparison_protocol.md"
])

expected_protocol = (
    "471d924e356dbafc1682fea5ecfb9916"
    "763af746089359986b59ff412ace56fa"
)

if protocol.read_bytes() != committed or sha(protocol) != expected_protocol:
    raise ValueError("Protocol differs from the committed version.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

for name, digest in manifest["input_sha256"].items():
    if sha(Path(name)) != digest:
        raise ValueError(f"QC input changed since fingerprinting: {name}")

expected_downloader = (
    "897e2e30bf66761fff03cc42f1e877e1"
    "1dad0a1789f766f507c55020d5254592"
)

if sha(downloader) != expected_downloader:
    raise ValueError("March/July downloader changed; stop and share the error.")

# Revalidate saved chunks; a missing file may be downloaded again.
config = runpy.run_path(str(downloader))

march_times = []
july_times = []
paths = []
grids = []

for start, end in config["PERIODS"]:
    path = Path("data/cyprus") / f"weather_48h_{start}_to_{end}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    response = saved["response"]

    times = pd.to_datetime(
        response["hourly"]["time"], utc=True, errors="raise"
    )

    if start < "2025-07-01":
        march_times.extend(times)
    else:
        july_times.extend(times)

    paths.append(path)
    grids.append({
        "file": str(path),
        "latitude": response["latitude"],
        "longitude": response["longitude"]
    })

march_times = pd.DatetimeIndex(march_times)
july_times = pd.DatetimeIndex(july_times)

expected_march = pd.date_range(
    "2025-02-28", "2025-04-01",
    freq="h", inclusive="left", tz="UTC"
)

expected_july = pd.date_range(
    "2025-07-22", "2025-07-31",
    freq="h", inclusive="left", tz="UTC"
)

if not march_times.equals(expected_march):
    raise ValueError("March weather timestamps are missing or duplicated.")

if not july_times.equals(expected_july):
    raise ValueError("July weather timestamps are missing or duplicated.")

march = pd.read_csv("reports/cyprus_2025_03_hourly_qc.csv")
march = march[march["status"] == "complete"].copy()
march["utc_end"] = pd.to_datetime(
    march["hour_end_utc"], utc=True, errors="raise"
)
march_match = march["utc_end"].isin(march_times)

july = pd.read_csv("reports/cyprus_2025_07_hourly_qc.csv")
july = july[july["status"] == "complete"].copy()

july["utc_start"] = pd.to_datetime(
    july["hour_start_utc"], utc=True, errors="raise"
)
july["utc_end"] = pd.to_datetime(
    july["hour_end_utc"], utc=True, errors="raise"
)
july["local_start"] = july["utc_start"].dt.tz_convert("Asia/Nicosia")

start = pd.Timestamp("2025-07-23", tz="Asia/Nicosia")
end = pd.Timestamp("2025-07-31", tz="Asia/Nicosia")

test = july[
    (july["local_start"] >= start)
    & (july["local_start"] < end)
]

history = pd.DatetimeIndex(july["utc_start"])

both = (
    (test["utc_start"] - pd.Timedelta(hours=72)).isin(history)
    & (test["utc_start"] - pd.Timedelta(days=7)).isin(history)
)

test_match = test["utc_end"].isin(july_times)

result = {
    "warning": (
        "Coverage only; no models trained or scored. May inputs must be "
        "rechecked during evaluation. Timezone and alignment are provisional, "
        "weather uses a grid point, and previous-day2 is not necessarily "
        "a simultaneous 48-hour forecast. Baselines assume timely telemetry."
    ),
    "march_complete_hours": len(march),
    "march_weather_matched": int(march_match.sum()),
    "march_weather_unmatched": int((~march_match).sum()),
    "july_holdout_calendar_hours": 192,
    "july_complete_target_hours": len(test),
    "july_weather_matched": int(test_match.sum()),
    "july_weather_unmatched": int((~test_match).sum()),
    "july_both_baselines_before_weather": int(both.sum()),
    "july_both_baselines_and_weather": int((both & test_match).sum()),
    "march_and_july_feasibility_floors_met": bool(
        int(march_match.sum()) >= 50
        and int((both & test_match).sum()) >= 20
    ),
    "may_105_original_training_rows_rechecked": False,
    "response_grid_points": grids,
    "protocol_sha256": sha(protocol),
    "qc_manifest_sha256": sha(manifest_path),
    "downloader_sha256": sha(downloader),
    "join_script_sha256": sha(Path("cyprus_july_weather_join.py")),
    "weather_input_sha256": {str(path): sha(path) for path in paths}
}

with out.open("x", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
