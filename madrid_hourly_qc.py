import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

source = Path("data/madrid/power_202504.zip")
reports = Path("reports")
reports.mkdir(exist_ok=True)

frames = []

with zipfile.ZipFile(source) as archive:
    names = sorted(
        name for name in archive.namelist()
        if name.endswith(".csv")
    )

    if len(names) != 6:
        raise ValueError("Expected six April CSV files. Verify the ZIP.")

    for name in names:
        with archive.open(name) as file:
            frame = pd.read_csv(
                file, usecols=["timestamp", "T1_S1_Pac"]
            )
        frame["source_file"] = name
        frames.append(frame)

data = pd.concat(frames, ignore_index=True)
data["time"] = pd.to_datetime(data["timestamp"], errors="coerce")
data["power_w"] = pd.to_numeric(data["T1_S1_Pac"], errors="coerce")

invalid_timestamps = int(data["time"].isna().sum())
valid = data.dropna(subset=["time"]).copy()

start = pd.Timestamp("2025-04-25 00:00:00")
end = pd.Timestamp("2025-05-01 00:00:00")

if ((valid["time"] < start) | (valid["time"] >= end)).any():
    raise ValueError("Timestamp outside the expected April range.")

valid["hour_start_local"] = valid["time"].dt.floor("h")
grouped = {
    hour: group
    for hour, group in valid.groupby("hour_start_local")
}

rows = []

for hour in pd.date_range(start, end, freq="h", inclusive="left"):
    group = grouped.get(hour)

    count = 0
    duplicates = 0
    conflicts = 0
    invalid_power = 0
    mean_w = None

    if group is None:
        status = "no_readings"
    else:
        count = len(group)
        duplicates = int(group["time"].duplicated().sum())
        conflicts = int(
            (group.groupby("time")["power_w"].nunique(dropna=False) > 1).sum()
        )
        invalid_power = int(
            (~np.isfinite(group["power_w"].to_numpy())).sum()
        )

        expected = pd.date_range(hour, periods=720, freq="5s")
        observed = pd.DatetimeIndex(group["time"]).sort_values()

        if conflicts:
            status = "conflicting_duplicates"
        elif duplicates:
            status = "duplicate_timestamps"
        elif invalid_power:
            status = "invalid_power"
        elif not observed.equals(expected):
            status = "incomplete_or_off_grid"
        else:
            status = "complete"
            mean_w = float(group["power_w"].mean())

    end_local = hour + pd.Timedelta(hours=1)
    end_utc = end_local.tz_localize("Europe/Madrid").tz_convert("UTC")

    rows.append({
        "hour_start_local": str(hour),
        "hour_end_local": str(end_local),
        "hour_end_utc": end_utc.isoformat(),
        "status": status,
        "raw_rows": count,
        "duplicate_rows": duplicates,
        "conflicting_timestamps": conflicts,
        "invalid_power_rows": invalid_power,
        "sampled_mean_power_w": mean_w
    })

hourly = pd.DataFrame(rows)
hourly_path = reports / "madrid_hourly_qc.csv"
hourly.to_csv(hourly_path, index=False)

summary = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "channel": "T1_S1_Pac",
    "input_rows": len(data),
    "invalid_timestamp_rows_excluded": invalid_timestamps,
    "total_hours_checked": len(hourly),
    "hours_by_status": hourly["status"].value_counts().to_dict(),
    "notes": (
        "Sampled mean power in watts is reported only for complete, "
        "unique and valid five-second grids. No interpolation, "
        "zero-filling or automatic duplicate resolution. Hour-end UTC "
        "is a provisional key for preceding-hour weather radiation."
    )
}

summary_path = reports / "madrid_qc_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print(json.dumps(summary, indent=2))
print(f"\nSaved: {hourly_path} and {summary_path}")
