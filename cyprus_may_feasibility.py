import hashlib
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

protocol = Path("reports/cyprus_may_protocol.md")
out = Path("reports/cyprus_may_feasibility.json")

if out.exists():
    raise FileExistsError(
        f"{out} already exists. Preserve it before rerunning."
    )

committed = subprocess.check_output(
    ["git", "show", "HEAD:reports/cyprus_may_protocol.md"]
)

if protocol.read_bytes() != committed:
    raise ValueError("Local protocol differs from the committed protocol.")

text = committed.decode("utf-8")

files = {
    "raw CSV": Path("data/cyprus/2025_05.csv"),
    "audit script": Path("cyprus_may_audit.py"),
    "hourly QC CSV": Path("reports/cyprus_may_hourly_qc.csv"),
    "audit summary": Path("reports/cyprus_may_qc_summary.json")
}

hashes = {}

for label, path in files.items():
    pattern = r"^- SHA-256 " + re.escape(label) + r": ([a-f0-9]{64})$"
    match = re.search(pattern, text, re.MULTILINE)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if not match or digest != match.group(1):
        raise ValueError(f"{path} does not match the committed protocol.")

    hashes[str(path)] = digest

qc = pd.read_csv(
    files["hourly QC CSV"],
    usecols=["hour_start_local", "status"]
)

qc["local_start"] = pd.to_datetime(
    qc["hour_start_local"], utc=True, errors="raise"
).dt.tz_convert("Asia/Nicosia")

expected = pd.date_range(
    "2025-05-01", "2025-06-01",
    freq="h", inclusive="left", tz="Asia/Nicosia"
)

if not pd.DatetimeIndex(qc["local_start"]).sort_values().equals(expected):
    raise ValueError("QC timestamps are missing or duplicated.")

complete = qc[qc["status"] == "complete"].copy()

cutoff = pd.Timestamp("2025-05-21 23:00", tz="Asia/Nicosia")
start = pd.Timestamp("2025-05-24", tz="Asia/Nicosia")
end = pd.Timestamp("2025-06-01", tz="Asia/Nicosia")

train = complete[complete["local_start"] < cutoff]
test = complete[
    (complete["local_start"] >= start)
    & (complete["local_start"] < end)
]

history = pd.DatetimeIndex(complete["local_start"])

has_72 = (
    test["local_start"] - pd.Timedelta(hours=72)
).isin(history)

has_7d = (
    test["local_start"] - pd.Timedelta(days=7)
).isin(history)

both = has_72 & has_7d

def days(frame):
    return (
        frame["local_start"]
        .dt.strftime("%Y-%m-%d")
        .value_counts()
        .sort_index()
        .to_dict()
    )

result = {
    "warning": (
        "Quality and baseline availability only. Weather has not been "
        "joined; baselines assume timely telemetry. No model has been "
        "trained or scored. Data quality was inspected before the protocol."
    ),
    "training_complete_hours_before_weather": len(train),
    "training_complete_hours_by_local_day": days(train),
    "holdout_calendar_hours": 192,
    "holdout_complete_hours_before_weather": len(test),
    "missing_72h_baseline": int((~has_72).sum()),
    "missing_7d_baseline": int((~has_7d).sum()),
    "holdout_with_both_baselines_before_weather": int(both.sum()),
    "paired_hours_by_local_day_before_weather": days(test[both]),
    "protocol_sha256": hashlib.sha256(committed).hexdigest(),
    "input_sha256": hashes
}

with out.open("x", encoding="utf-8") as file:
    json.dump(result, file, indent=2)

print(json.dumps(result, indent=2))
print(f"\nSaved: {out}")
