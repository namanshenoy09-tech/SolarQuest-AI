import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

archive_path = Path("data/madrid/power_202506.zip")
reports = Path("reports")
reports.mkdir(exist_ok=True)

csv_path = reports / "madrid_june_hourly_qc.csv"
json_path = reports / "madrid_june_qc_summary.json"

archive_bytes = archive_path.read_bytes()
expected_md5 = "0b0080caf7b256c297a592bedf11385d"

if hashlib.md5(archive_bytes).hexdigest() != expected_md5:
    raise ValueError("June archive checksum does not match Zenodo.")

expected_files = {
    f"power_202506{day:02d}.csv"
    for day in range(1, 31)
}

frames = []

with zipfile.ZipFile(archive_path) as archive:
    names = sorted(
        name for name in archive.namelist()
        if name.endswith(".csv")
    )

    if not names:
        raise ValueError("No daily CSV files found.")

    if len(names) != len(set(names)):
        raise ValueError("Duplicate CSV filenames in archive.")

    if not set(names).issubset(expected_files):
        raise ValueError("Unexpected CSV filename in archive.")

    missing_files = sorted(expected_files - set(names))

    for name in names:
        with archive.open(name) as file:
            frame = pd.read_csv(
                file, usecols=["timestamp", "T1_S1_Pac"]
            )
        frames.append(frame)

data = pd.concat(frames, ignore_index=True)

data["time"] = pd.to_datetime(
    data["timestamp"], errors="coerce"
)
data["power_w"] = pd.to_numeric(
    data["T1_S1_Pac"], errors="coerce"
)

invalid_timestamps = int(data["time"].isna().sum())
valid = data.dropna(subset=["time"]).copy()

start = pd.Timestamp("2025-06-01")
end = pd.Timestamp("2025-07-01")

if ((valid["time"] < start) | (valid["time"] >= end)).any():
    raise ValueError("Timestamp outside June.")

valid["hour_start_local"] = valid["time"].dt.floor("h")
groups = {
    hour: group
    for hour, group in valid.groupby("hour_start_local")
}

rows = []

for hour in pd.date_range(start, end, freq="h", inclusive="left"):
    group = groups.get(hour)
    count = duplicates = conflicts = invalid_power = 0
    mean_w = None

    if group is None:
        status = "no_readings"
    else:
        count = len(group)
        duplicates = int(group["time"].duplicated().sum())
        conflicts = int(
            (
                group.groupby("time")["power_w"]
                .nunique(dropna=False) > 1
            ).sum()
        )
        invalid_power = int(
            (~np.isfinite(group["power_w"].to_numpy())).sum()
        )

        expected_grid = pd.date_range(
            hour, periods=720, freq="5s"
        )
        observed = pd.DatetimeIndex(group["time"]).sort_values()

        if conflicts:
            status = "conflicting_duplicates"
        elif duplicates:
            status = "duplicate_timestamps"
        elif invalid_power:
            status = "invalid_power"
        elif not observed.equals(expected_grid):
            status = "incomplete_or_off_grid"
        else:
            status = "complete"
            mean_w = float(group["power_w"].mean())

    end_local = hour + pd.Timedelta(hours=1)
    end_utc = end_local.tz_localize(
        "Europe/Madrid"
    ).tz_convert("UTC")

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
hourly.to_csv(csv_path, index=False)

complete = set(pd.to_datetime(
    hourly.loc[hourly["status"] == "complete", "hour_start_local"]
))

test_hours = pd.date_range(
    "2025-06-23", "2025-07-01", freq="h", inclusive="left"
)

targets = [hour for hour in test_hours if hour in complete]

paired = [
    hour for hour in targets
    if hour - pd.Timedelta(hours=72) in complete
    and hour - pd.Timedelta(days=7) in complete
]

summary = {
    "source": str(archive_path),
    "source_sha256": hashlib.sha256(archive_bytes).hexdigest(),
    "channel": "T1_S1_Pac",
    "daily_files_in_archive": len(names),
    "missing_daily_files": missing_files,
    "input_rows": len(data),
    "invalid_timestamp_rows_excluded": invalid_timestamps,
    "total_hours_checked": len(hourly),
    "hours_by_status": hourly["status"].value_counts().to_dict(),
    "fixed_june_23_30_holdout": {
        "total_calendar_hours": len(test_hours),
        "complete_target_hours": len(targets),
        "complete_targets_with_both_baselines": len(paired)
    },
    "notes": (
        "Sampled mean power in watts is provided only for complete, "
        "unique, valid five-second grids. No interpolation, zero-fill "
        "or automatic duplicate resolution. Hour-end UTC is a "
        "provisional alignment key. Counts do not check weather."
    )
}

json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nSaved: {csv_path} and {json_path}")
