"""UTC-grid quality audit for the pre-registered Cyprus files."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

months = {
    "2025_03": (
        "2025-03-01", "2025-04-01",
        "78e899e964dd46b549459e9c7481bb04"
    ),
    "2025_07": (
        "2025-07-01", "2025-08-01",
        "22f5db59ca74c8b5d2a8aef5f56065e2"
    )
}

if len(sys.argv) != 2 or sys.argv[1] not in months:
    raise SystemExit(
        "Usage: python cyprus_multi_month_qc.py 2025_03 or 2025_07"
    )

month = sys.argv[1]
start_date, end_date, expected_md5 = months[month]

source = Path(f"data/cyprus/{month}.csv")
protocol = Path("reports/cyprus_july_training_comparison_protocol.md")
report_csv = Path(f"reports/cyprus_{month}_hourly_qc.csv")
report_json = Path(f"reports/cyprus_{month}_qc_summary.json")

if report_csv.exists() or report_json.exists():
    raise FileExistsError(
        "Monthly report already exists; preserve it before rerunning."
    )

committed = subprocess.check_output([
    "git", "show",
    "HEAD:reports/cyprus_july_training_comparison_protocol.md"
])

expected_protocol_sha = (
    "471d924e356dbafc1682fea5ecfb9916"
    "763af746089359986b59ff412ace56fa"
)

if (
    protocol.read_bytes() != committed
    or hashlib.sha256(committed).hexdigest() != expected_protocol_sha
):
    raise ValueError("The fixed July protocol differs from the committed version.")

blob = source.read_bytes()

if hashlib.md5(blob).hexdigest() != expected_md5:
    raise ValueError(f"{source} does not match the publisher checksum.")

raw = pd.read_csv(source, usecols=["datetime", "pv_power (W)"])
text = raw["datetime"].astype("string")

has_offset = text.str.contains(
    r"(?:Z|[+-]\d{2}:\d{2})$", regex=True, na=False
)

times = pd.to_datetime(
    text.where(has_offset), utc=True, errors="coerce"
)

if times.isna().any():
    raise ValueError(
        f"{int(times.isna().sum())} timestamps are invalid or lack offsets."
    )

power = pd.to_numeric(raw["pv_power (W)"], errors="coerce")
local = times.dt.tz_convert("Asia/Nicosia")

raw_offsets = (
    text.str.extract(r"(Z|[+-]\d{2}:\d{2})$", expand=False)
    .replace("Z", "+00:00")
    .str.replace(":", "", regex=False)
)

if not raw_offsets.eq(local.dt.strftime("%z")).fillna(False).all():
    raise ValueError(
        "Source offsets differ from the provisional Asia/Nicosia timezone."
    )

start = pd.Timestamp(start_date, tz="Asia/Nicosia").tz_convert("UTC")
end = pd.Timestamp(end_date, tz="Asia/Nicosia").tz_convert("UTC")

if ((times < start) | (times >= end)).any():
    raise ValueError("Timestamp outside the expected local month.")

valid = pd.DataFrame({"utc_time": times, "power_w": power})
valid["utc_hour"] = valid["utc_time"].dt.floor("h")

groups = {
    hour: group
    for hour, group in valid.groupby("utc_hour")
}

rows = []

for utc_hour in pd.date_range(start, end, freq="h", inclusive="left"):
    group = groups.get(utc_hour)
    count = duplicates = invalid = 0
    mean_w = None

    if group is None:
        status = "no_readings"
    else:
        count = len(group)
        duplicates = int(group["utc_time"].duplicated().sum())
        invalid = int((~np.isfinite(group["power_w"].to_numpy())).sum())

        grid = pd.date_range(utc_hour, periods=120, freq="30s")
        observed = pd.DatetimeIndex(group["utc_time"]).sort_values()

        if duplicates:
            status = "duplicate_timestamps"
        elif not observed.equals(grid):
            status = "incomplete_or_off_grid"
        elif invalid:
            status = "missing_or_invalid_power"
        else:
            status = "complete"
            mean_w = float(group["power_w"].mean())

    rows.append({
        "hour_start_local": utc_hour.tz_convert("Asia/Nicosia").isoformat(),
        "hour_start_utc": utc_hour.isoformat(),
        "hour_end_utc": (utc_hour + pd.Timedelta(hours=1)).isoformat(),
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
    "protocol_sha256": hashlib.sha256(committed).hexdigest(),
    "audit_script_sha256": hashlib.sha256(
        Path("cyprus_multi_month_qc.py").read_bytes()
    ).hexdigest(),
    "month": month,
    "input_rows": len(raw),
    "missing_power_rows": int(raw["pv_power (W)"].isna().sum()),
    "invalid_nonmissing_power_rows": int(
        (raw["pv_power (W)"].notna() & ~np.isfinite(power)).sum()
    ),
    "finite_power_rows": int(np.isfinite(power).sum()),
    "negative_power_rows": int((power < 0).sum()),
    "source_offsets": text.str.extract(
        r"(Z|[+-]\d{2}:\d{2})$", expand=False
    ).value_counts().to_dict(),
    "total_local_hours_checked": len(hourly),
    "hours_by_status": hourly["status"].value_counts().to_dict(),
    "notes": (
        "UTC-grid quality audit only; no model trained or scored. "
        "A complete hour has 120 unique, aligned, finite 30-second "
        "power readings. Missing is not zero. Asia/Nicosia and hour-end "
        "alignment remain provisional; March has a local clock change."
    )
}

if month == "2025_07":
    good = hourly[hourly["status"] == "complete"].copy()
    good["utc_start"] = pd.to_datetime(
        good["hour_start_utc"], utc=True, errors="raise"
    )
    good["local_start"] = good["utc_start"].dt.tz_convert("Asia/Nicosia")

    test = good[
        (good["local_start"] >= pd.Timestamp("2025-07-23", tz="Asia/Nicosia"))
        & (good["local_start"] < pd.Timestamp("2025-07-31", tz="Asia/Nicosia"))
    ]

    history = pd.DatetimeIndex(good["utc_start"])

    has_72 = (
        test["utc_start"] - pd.Timedelta(hours=72)
    ).isin(history)

    has_7d = (
        test["utc_start"] - pd.Timedelta(days=7)
    ).isin(history)

    summary["fixed_july_23_30_holdout"] = {
        "calendar_hours": 192,
        "complete_target_hours": len(test),
        "missing_72h_baseline": int((~has_72).sum()),
        "missing_7d_baseline": int((~has_7d).sum()),
        "complete_targets_with_both_baselines_before_weather": int(
            (has_72 & has_7d).sum()
        )
    }

report_csv.parent.mkdir(exist_ok=True)
hourly.to_csv(report_csv, index=False)
report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print(f"\nSaved: {report_csv} and {report_json}")
