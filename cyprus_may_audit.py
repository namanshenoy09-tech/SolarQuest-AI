import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

source = Path("data/cyprus/2025_05.csv")
reports = Path("reports")
reports.mkdir(exist_ok=True)

output_csv = reports / "cyprus_may_hourly_qc.csv"
output_json = reports / "cyprus_may_qc_summary.json"

if output_csv.exists() or output_json.exists():
    raise FileExistsError(
        "Cyprus May report already exists. Preserve it before rerunning."
    )

blob = source.read_bytes()

if hashlib.md5(blob).hexdigest() != "02771a68ce20c5c2d3d9a0952c0198ed":
    raise ValueError("Source checksum does not match Zenodo.")

raw = pd.read_csv(source, usecols=["datetime", "pv_power (W)"])

text = raw["datetime"].astype("string")
has_offset = text.str.contains(
    r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False
)

times = pd.to_datetime(text.where(has_offset), utc=True, errors="coerce")
power = pd.to_numeric(raw["pv_power (W)"], errors="coerce")

missing_times = int(times.isna().sum())
missing_power = int(raw["pv_power (W)"].isna().sum())
invalid_power = int(
    (raw["pv_power (W)"].notna() & ~np.isfinite(power)).sum()
)

valid = pd.DataFrame({
    "utc_time": times,
    "power_w": power
}).dropna(subset=["utc_time"]).copy()

valid["local_time"] = valid["utc_time"].dt.tz_convert("Asia/Nicosia")

start = pd.Timestamp("2025-05-01", tz="Asia/Nicosia")
end = pd.Timestamp("2025-06-01", tz="Asia/Nicosia")

if ((valid["local_time"] < start) | (valid["local_time"] >= end)).any():
    raise ValueError("Timestamp outside expected local May range.")

if missing_times:
    raise ValueError(
        f"{missing_times} missing, invalid or offset-free timestamps. "
        "Stop and investigate."
    )

raw_offsets = (
    text.str.extract(r"(Z|[+-]\d{2}:\d{2})$", expand=False)
    .replace("Z", "+00:00")
    .str.replace(":", "", regex=False)
)

local_offsets = valid["local_time"].dt.strftime("%z")

if not raw_offsets.eq(local_offsets).fillna(False).all():
    raise ValueError(
        "Source offsets do not match the provisional Asia/Nicosia timezone."
    )

valid["local_hour"] = valid["local_time"].dt.floor("h")
groups = {
    hour: group
    for hour, group in valid.groupby("local_hour")
}

rows = []

for hour in pd.date_range(start, end, freq="h", inclusive="left"):
    group = groups.get(hour)
    count = duplicates = invalid = 0
    mean_w = None

    if group is None:
        status = "no_readings"
    else:
        count = len(group)
        duplicates = int(group["utc_time"].duplicated().sum())
        invalid = int(
            (~np.isfinite(group["power_w"].to_numpy())).sum()
        )

        expected = pd.date_range(hour, periods=120, freq="30s")
        observed = pd.DatetimeIndex(group["local_time"]).sort_values()

        if duplicates:
            status = "duplicate_timestamps"
        elif not observed.equals(expected):
            status = "incomplete_or_off_grid"
        elif invalid:
            status = "missing_or_invalid_power"
        else:
            status = "complete"
            mean_w = float(group["power_w"].mean())

    rows.append({
        "hour_start_local": hour.isoformat(),
        "hour_end_utc": (
            hour + pd.Timedelta(hours=1)
        ).tz_convert("UTC").isoformat(),
        "status": status,
        "raw_rows": count,
        "duplicate_rows": duplicates,
        "missing_or_invalid_power_rows": invalid,
        "sampled_mean_power_w": mean_w
    })

hourly = pd.DataFrame(rows)

summary = {
    "source": str(source),
    "source_sha256": hashlib.sha256(blob).hexdigest(),
    "dataset": "https://doi.org/10.5281/zenodo.20800202",
    "channel": "pv_power (W)",
    "input_rows": len(raw),
    "missing_invalid_or_offset_free_timestamps": missing_times,
    "missing_power_rows": missing_power,
    "invalid_nonmissing_power_rows": invalid_power,
    "finite_power_rows": int(np.isfinite(power).sum()),
    "negative_power_rows": int((power < 0).sum()),
    "total_local_hours_checked": len(hourly),
    "hours_by_status": hourly["status"].value_counts().to_dict(),
    "notes": (
        "Exploratory quality audit only; no training or forecast evaluation. "
        "Asia/Nicosia is a provisional timezone interpretation checked "
        "against source offsets for May only. Power is labelled W, but "
        "sampling and interval semantics and capacity definitions remain "
        "to be confirmed. Sampled mean is provided only when all 120 "
        "unique on-grid readings are finite. Missing is not zero."
    )
}

hourly.to_csv(output_csv, index=False)
output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nSaved: {output_csv} and {output_json}")
