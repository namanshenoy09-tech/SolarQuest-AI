# SolarQuest-AI upgrade — step-by-step

## What this package is

A proposed upgrade built against the public repository snapshot at commit `32aa21a` (the clone inspected for this work). The checked-in `app.py` at that snapshot is **V1**; V2 and V3 are present as notebook cells. The V4 code in the project documentation was not in the checked-in `app.py`. This package **replaces the V1 app** with a safer research prototype, rather than patching a V4 repository file. It does not modify your GitHub repository automatically.

The inspected notebook output reports **3,157 merged plant timestamps**, and a random-split R² printed as **99.38%**; your supplied documentation mentions 68,000+ timestamps and R² 0.9838. Those figures are not reconciled. Do not use either figure as a 48-hour forecast-accuracy claim. Raw inverter rows and merged timestamps are different quantities. The original notebook is retained for historical reference, not as the source of current product claims; avoid running its `%%writefile app.py` cells, which overwrite the new app.

## Before you begin

- Install Git and Python 3.11. Open a terminal (macOS/Linux Terminal or Windows PowerShell).
- You need a local clone of `github.com/namanshenoy09-tech/SolarQuest-AI` and permission to push to it if you want to publish changes.
- Download and unzip this package. It contains **code only**, not the trained model or Kaggle data. Keep your original repository and files backed up.

## 1. Make a branch and copy files

In a terminal, enter your existing repository directory. If you haven't cloned it, run:

```bash
git clone https://github.com/namanshenoy09-tech/SolarQuest-AI.git
cd SolarQuest-AI
```

If already cloned, run `cd` into that existing folder instead. Create a branch:

```bash
git switch -c prototype-reliability-upgrade
```

From the extracted upgrade folder, copy **`app.py`, `solarquest_core.py`, `train_evaluate.py`, `requirements.txt`, `README.md`, `APPLY_CHANGES.md`, and the entire `tests` folder** into the root of your repository. Approve replacing `app.py` and `README.md`. Keep your existing `SolarQuest.ipynb`, `LICENSE` and trusted `solar_ai_model.pkl` in place. The new README deliberately replaces inaccurate production claims.

## 2. Create an isolated environment and install packages

macOS / Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python directly (`.\.venv\Scripts\python.exe`) instead of `python` in the following commands. If your model was saved with an incompatible scikit-learn version, retrain it with this environment and the original training data; do not load a model you don't trust.

## 3. Test and run

```bash
python -m pytest -q
python -m streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. Choose Manual simulation, Weather forecast experiment, or Upload hourly telemetry in the sidebar. The weather experiment needs an internet connection and the **actual training plant** location/time zone. The displayed coordinates are placeholders; they are not known training-site coordinates. Confirm weather and dataset feature units before interpreting output.

## 4. Evaluate and retrain using the original CSVs

Place the Kaggle files `Plant_1_Generation_Data.csv` and `Plant_1_Weather_Sensor_Data.csv` in a local `data/` folder. The source dataset is not included; obtain it under its own license and check column and unit definitions. Do not commit private plant data.

```bash
python train_evaluate.py --generation data/Plant_1_Generation_Data.csv --weather data/Plant_1_Weather_Sensor_Data.csv --expected-inverters 22
```

`22` is an example based on the original notebook's description; use it only if confirmed for your plant and period. The script rejects duplicate timestamps/sources and drops timestamp totals with a different inverter count. Matching a count alone does not prove all correct devices reported. If you omit the option, partially reported intervals may be included. Optionally add `--capacity-kw 25000` **only if that capacity is verified**; this adds normalized MAE and does not scale the plant's predictions. You may also add `--daylight-threshold 0.01` to report a daylight-only MAE, but choose that threshold only after verifying the irradiation units and an appropriate cutoff.

The command writes `evaluation_report.json` and `solar_ai_model_retrained.pkl`, without overwriting the original model. It reports three expanding chronological test folds. The final saved model is trained on the first 80% of the records, leaving the last 20% as its test fold. The prior-day baseline is scored where an exact prior-day observation exists; compare it to the model MAE on **the same baseline rows** in each fold. Training features are actual historical sensor readings, not archived weather forecasts, so these results are **not a 48-hour forecast backtest**.

After examining the report and checking units, you can use the new model in the app by backing up the old `solar_ai_model.pkl` and renaming/copying `solar_ai_model_retrained.pkl` to `solar_ai_model.pkl` in the repository root. Restart Streamlit. Preserve the report and environment versions so you can reproduce the run.

## 5. Try the telemetry upload

Create a CSV, e.g. `example_telemetry.csv`:

```csv
timestamp,actual_power_kw,expected_power_kw
2025-01-01T10:00:00Z,75,100
2025-01-01T11:00:00Z,80,105
```

Each timestamp is the UTC-offset-aware **start of a completed one-hour interval** and each power number is the interval's average kW. Every row must start on an exact UTC hour. Upload it under **Upload hourly telemetry**. Here the total shortfall is 50 kWh and at $0.06/kWh the illustrative value is $3.00. The app sums only positive hourly shortfalls, does not offset surplus intervals, does not fill missing hours, and cannot distinguish faults from curtailment or forecast error. Never share or upload confidential data without authorization.

## 6. Review and publish when ready

```bash
git status
git diff -- app.py README.md
git add app.py solarquest_core.py train_evaluate.py requirements.txt README.md APPLY_CHANGES.md tests
git commit -m "Add honest prototype and chronological evaluation"
git push -u origin prototype-reliability-upgrade
```

Open a pull request on GitHub and review it before merging. `git diff` does not display the contents of new untracked files; open those files too. Do not commit plant CSVs, private data, secrets or model artifacts unless intended. If Git asks you to authenticate, follow GitHub's current authentication instructions.

## Next work requiring data or product decisions

1. Verify sensor units, capacity, site identity, inverter coverage and model provenance.
2. Obtain archived weather **forecasts issued at known times** and matching future actual generation; backtest by 1–48-hour lead, daylight and weather regime against appropriate baselines. The current script cannot establish forecast performance.
3. Add real, permissioned SCADA ingestion with authentication, access controls, monitoring and agreed data-retention terms. Validate and classify faults using independently confirmed maintenance events before using diagnostic labels.
4. Interview operators, agree pilot success criteria, and measure realized outcomes before pricing or ROI claims. Add tests against real anonymized data and independently review the model before operational use.
