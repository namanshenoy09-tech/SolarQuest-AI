"""Registered same-site Cyprus training comparison, not operational validation."""
import contextlib
import hashlib
import io
import json
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def verify(d):
    for p, digest in d.items():
        if sha(p) != digest:
            raise ValueError(f'Input has changed: {p}')

out = Path('reports/cyprus_july_training_comparison.json')
if out.exists():
    raise FileExistsError(f'{out} exists; preserve the original result.')
protocol = Path('reports/cyprus_july_training_comparison_protocol.md')
if sha(protocol) != '471d924e356dbafc1682fea5ecfb9916763af746089359986b59ff412ace56fa':
    raise ValueError('July protocol hash differs.')
if protocol.read_bytes() != subprocess.check_output(['git', 'show', 'HEAD:reports/cyprus_july_training_comparison_protocol.md']):
    raise ValueError('July protocol differs from committed copy.')
original_path = Path('reports/cyprus_may_holdout.json')
if sha(original_path) != '4f0081ad2892ff40a58fbdf9aae64a73b82134edb9e5d4407f3c73381251d50c':
    raise ValueError('May report hash differs.')
original = read(original_path)
if sklearn.__version__ != original['sklearn_version'] or sys.version.split()[0] != original['python_version']:
    raise ValueError('Python or scikit-learn version differs from May.')
verify(original['input_sha256'])
reproduction = read('reports/cyprus_may_reproduction.json')
if reproduction != original:
    raise ValueError('May reproduction differs from the original report.')
join = read('reports/cyprus_july_weather_join.json')
manifest_path = Path('reports/cyprus_july_qc_manifest.json')
if join['protocol_sha256'] != sha(protocol) or join['qc_manifest_sha256'] != sha(manifest_path):
    raise ValueError('July join provenance differs.')
verify(read(manifest_path)['input_sha256'])
verify(join['weather_input_sha256'])
downloader = Path('fetch_cyprus_march_july_weather.py')
if sha(downloader) != join['downloader_sha256'] or sha('cyprus_july_weather_join.py') != join['join_script_sha256']:
    raise ValueError('July downloader or join script changed.')
# The downloader validates requests, units, all timestamps and numerical values.
# All files were hash-checked above, so this does not fetch new data.
with contextlib.redirect_stdout(io.StringIO()):
    config = runpy.run_path(str(downloader))

# Refit the registered May model with the original evaluator; verify all May
# report fields before using its training frame and fitted model.
evaluator = Path('evaluate_cyprus_may.py')
if sha(evaluator) != '4a6fcc7afcd64eb27aae2a22ab071d60a7a23ac6be8e1fafcdb458f2b00b892c':
    raise ValueError('May evaluator differs from the registered version.')
text = evaluator.read_text(encoding='utf-8')
old = 'out = Path("reports/cyprus_may_holdout.json")'
if text.count(old) != 1:
    raise ValueError('Unexpected May evaluator structure.')
with tempfile.TemporaryDirectory(prefix='cyprus_may_check_', dir='reports') as tmp:
    temp_report = Path(tmp) / 'result.json'
    modified = text.replace(old, f'out = Path({str(temp_report)!r})')
    namespace = {'__name__': '__main__'}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(modified, str(evaluator), 'exec'), namespace)
    if read(temp_report) != original:
        raise ValueError('The current May refit does not reproduce the saved report.')
may_train = namespace['train'].copy()
model_a = namespace['model']
features = namespace['features']
if len(may_train) != 105:
    raise ValueError('May training rows do not equal 105.')

parts = []
for start, end in config['PERIODS']:
    path = Path('data/cyprus') / f'weather_48h_{start}_to_{end}.json'
    hourly = read(path)['response']['hourly']
    parts.append(pd.DataFrame({
        'utc_end': pd.to_datetime(hourly['time'], utc=True, errors='raise'),
        'temperature': hourly[config['FIELDS'][0]],
        'radiation': hourly[config['FIELDS'][1]]
    }))
weather = pd.concat(parts, ignore_index=True)
if weather['utc_end'].duplicated().any():
    raise ValueError('Duplicate weather timestamps.')

def complete_month(month):
    qc = pd.read_csv(f'reports/cyprus_{month}_hourly_qc.csv')
    qc = qc[qc['status'] == 'complete'].copy()
    qc['local_start'] = pd.to_datetime(qc['hour_start_local'], utc=True).dt.tz_convert('Asia/Nicosia')
    qc['utc_end'] = pd.to_datetime(qc['hour_end_utc'], utc=True)
    qc['power_w'] = pd.to_numeric(qc['sampled_mean_power_w'], errors='raise')
    if not np.isfinite(qc['power_w']).all() or qc['local_start'].duplicated().any():
        raise ValueError(f'Invalid {month} complete power or timestamps.')
    return qc

def join_weather(frame):
    merged = frame.merge(weather, on='utc_end', how='left', validate='one_to_one').sort_values('local_start')
    merged = merged.dropna(subset=['temperature', 'radiation']).copy()
    hour = merged['local_start'].dt.hour
    merged['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    merged['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    return merged

march = complete_month('2025_03')
march = march[(march['local_start'] >= pd.Timestamp('2025-03-01', tz='Asia/Nicosia')) & (march['local_start'] < pd.Timestamp('2025-04-01', tz='Asia/Nicosia'))]
march_train = join_weather(march)
july = complete_month('2025_07')
start = pd.Timestamp('2025-07-23', tz='Asia/Nicosia')
end = pd.Timestamp('2025-07-31', tz='Asia/Nicosia')
targets = july[(july['local_start'] >= start) & (july['local_start'] < end)]
test = join_weather(targets)
history = july.set_index('local_start')['power_w']
for name, hours in [('baseline_72h', 72), ('baseline_7d', 168)]:
    test[name] = history.reindex(pd.DatetimeIndex(test['local_start'] - pd.Timedelta(hours=hours))).to_numpy()
paired = test.dropna(subset=['baseline_72h', 'baseline_7d']).copy()
if (len(march_train) != join['march_weather_matched'] or len(test) != join['july_weather_matched'] or len(paired) != join['july_both_baselines_and_weather']):
    raise ValueError('Coverage differs from the saved join report.')
if len(march_train) < 50 or len(paired) < 20:
    raise ValueError('Registered feasibility floors not met; do not change the split.')

model_b = LinearRegression().fit(pd.concat([may_train[features], march_train[features]], ignore_index=True), pd.concat([may_train['power_w'], march_train['power_w']], ignore_index=True))
for name, model in [('model_a', model_a), ('model_b', model_b)]:
    paired[name] = np.where(paired['radiation'] <= 0, 0, np.maximum(0, model.predict(paired[features])))

def score(frame):
    if frame.empty:
        return None
    values = {'rows': len(frame)}
    for name in ['model_a', 'model_b', 'baseline_72h', 'baseline_7d']:
        values[name + '_mae_w'] = round(float(mean_absolute_error(frame['power_w'], frame[name])), 3)
    return values

daily = {}
for day in pd.date_range(start, end, freq='D', inclusive='left'):
    date = day.date()
    daily[str(date)] = {'calendar_hours': 24, 'complete_target_hours': int((targets['local_start'].dt.date == date).sum()), 'weather_matched_hours': int((test['local_start'].dt.date == date).sum()), 'paired_scores': score(paired[paired['local_start'].dt.date == date])}
inputs = [protocol, original_path, Path('reports/cyprus_may_reproduction.json'), manifest_path, Path('reports/cyprus_july_weather_join.json'), downloader, Path('cyprus_july_weather_join.py'), Path('evaluate_cyprus_july_comparison.py')]
inputs += [Path(p) for p in original['input_sha256']] + [Path(p) for p in read(manifest_path)['input_sha256']] + [Path(p) for p in join['weather_input_sha256']]
result = {'warning': 'Exploratory same-site retraining, not cross-site transfer or operational validation. Missing is not zero. Meter timing, capacity basis and timezone are provisional; archived previous-day2 is not necessarily one simultaneously issued 48-hour forecast. Baselines assume timely telemetry. No financial savings inferred.', 'python_version': sys.version.split()[0], 'sklearn_version': sklearn.__version__, 'may_report_reproduced_before_july_scoring': True, 'model_a_may_training_rows': len(may_train), 'model_b_may_training_rows': len(may_train), 'model_b_march_training_rows': len(march_train), 'model_b_total_training_rows': len(may_train) + len(march_train), 'july_holdout_start_local': str(start), 'july_holdout_end_exclusive_local': str(end), 'july_calendar_hours': 192, 'july_complete_targets': len(targets), 'july_weather_matched': len(test), 'missing_72h_baseline': int(test['baseline_72h'].isna().sum()), 'missing_7d_baseline': int(test['baseline_7d'].isna().sum()), 'paired_all': score(paired), 'paired_positive_forecast_radiation': score(paired[paired['radiation'] > 0]), 'daily_paired': daily, 'weather_request_and_grid_metadata': [{'file': str(Path('data/cyprus') / f'weather_48h_{s}_to_{e}.json'), 'request_parameters': read(Path('data/cyprus') / f'weather_48h_{s}_to_{e}.json')['request_parameters'], 'grid_latitude': read(Path('data/cyprus') / f'weather_48h_{s}_to_{e}.json')['response']['latitude'], 'grid_longitude': read(Path('data/cyprus') / f'weather_48h_{s}_to_{e}.json')['response']['longitude']} for s, e in config['PERIODS']], 'input_sha256': {str(p): sha(p) for p in inputs}}
with out.open('x', encoding='utf-8') as file:
    json.dump(result, file, indent=2)
print(json.dumps(result, indent=2))
print(f'\nSaved: {out}')
