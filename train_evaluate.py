"""Chronological retrospective benchmark. This is NOT a 48-hour-ahead evaluation."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from solarquest_core import FEATURES


def load_data(gen_path, weather_path, expected_inverters=None):
    gen = pd.read_csv(gen_path)
    weather = pd.read_csv(weather_path)
    for df, needed, name in [
        (gen, ['DATE_TIME','SOURCE_KEY','AC_POWER'], 'generation'),
        (weather, ['DATE_TIME','IRRADIATION','AMBIENT_TEMPERATURE','MODULE_TEMPERATURE'], 'weather')]:
        missing = sorted(set(needed) - set(df.columns))
        if missing: raise ValueError(f'{name} file missing columns: {missing}')
        if 'PLANT_ID' in df and df['PLANT_ID'].nunique(dropna=False) != 1:
            raise ValueError(f'{name} file must contain exactly one plant ID.')
    if 'PLANT_ID' in gen and 'PLANT_ID' in weather and str(gen['PLANT_ID'].iloc[0]) != str(weather['PLANT_ID'].iloc[0]):
        raise ValueError('Generation and weather PLANT_ID do not match.')
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'], dayfirst=True, errors='raise')
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'], errors='raise')
    if gen['DATE_TIME'].isna().any() or weather['DATE_TIME'].isna().any(): raise ValueError('Missing timestamps.')
    if gen['SOURCE_KEY'].isna().any(): raise ValueError('Missing inverter SOURCE_KEY.')
    if gen.duplicated(['DATE_TIME', 'SOURCE_KEY']).any(): raise ValueError('Duplicate timestamp/inverter rows.')
    if weather['DATE_TIME'].duplicated().any(): raise ValueError('Duplicate weather timestamps; select one weather sensor first.')
    gen['AC_POWER'] = pd.to_numeric(gen['AC_POWER'], errors='coerce')
    if not np.isfinite(gen['AC_POWER']).all() or (gen['AC_POWER'] < 0).any(): raise ValueError('Invalid generation power values.')
    for col in FEATURES[:-1]: weather[col] = pd.to_numeric(weather[col], errors='coerce')
    if not np.isfinite(weather[FEATURES[:-1]].to_numpy()).all(): raise ValueError('Invalid weather feature values.')
    if (weather['IRRADIATION'] < 0).any(): raise ValueError('Negative irradiation values.')
    gen_total = gen.groupby('DATE_TIME').agg(AC_POWER=('AC_POWER','sum'), inverter_count=('SOURCE_KEY','nunique')).reset_index()
    if expected_inverters is not None:
        if expected_inverters <= 0: raise ValueError('Expected inverter count must be positive.')
        before = len(gen_total)
        gen_total = gen_total[gen_total['inverter_count'] == expected_inverters]
        print(f'Excluded {before - len(gen_total)} generation timestamps without exactly {expected_inverters} inverters.')
    else:
        print('WARNING: no expected inverter count supplied; partial plant totals may remain.')
    df = gen_total.merge(weather[['DATE_TIME']+FEATURES[:-1]], on='DATE_TIME', how='inner', validate='one_to_one').sort_values('DATE_TIME').reset_index(drop=True)
    df['HOUR'] = df['DATE_TIME'].dt.hour
    if len(df) < 30: raise ValueError('Fewer than 30 usable merged timestamps; inspect data.')
    print(f'Generation rows: {len(gen):,}; matched plant timestamps: {len(df):,}.')
    return df


def evaluate(df, capacity_kw=None, daylight_threshold=None):
    if daylight_threshold is not None and (not np.isfinite(daylight_threshold) or daylight_threshold < 0):
        raise ValueError('Daylight threshold must be non-negative.')
    if capacity_kw is not None and (not np.isfinite(capacity_kw) or capacity_kw <= 0): raise ValueError('Capacity must be positive.')
    # Each test block is strictly later than its training block. Boundaries apply to timestamps, not random rows.
    fractions = [(0.4,0.6), (0.6,0.8), (0.8,1.0)]
    historical = df.set_index('DATE_TIME')['AC_POWER']
    reports=[]
    for fold,(a,b) in enumerate(fractions, 1):
        start,end=int(len(df)*a),int(len(df)*b)
        train,test=df.iloc[:start],df.iloc[start:end]
        model=LinearRegression().fit(train[FEATURES], train['AC_POWER'])
        pred=np.maximum(0, model.predict(test[FEATURES]))
        y=test['AC_POWER'].to_numpy()
        # Exact previous-day observations; baseline is only scored where available.
        prior = historical.reindex(pd.DatetimeIndex(test['DATE_TIME'] - pd.Timedelta(days=1))).to_numpy()
        valid = np.isfinite(prior)
        report = {'fold':fold, 'train_rows':len(train), 'test_rows':len(test),
                  'train_end':str(train['DATE_TIME'].iloc[-1]), 'test_start':str(test['DATE_TIME'].iloc[0]),
                  'test_end':str(test['DATE_TIME'].iloc[-1]),
                  'mae_kw':float(mean_absolute_error(y,pred)), 'rmse_kw':float(np.sqrt(mean_squared_error(y,pred))),
                  'r2':float(r2_score(y,pred)), 'baseline_matched_rows':int(valid.sum())}
        if valid.any():
            report['previous_day_mae_kw'] = float(mean_absolute_error(y[valid], prior[valid]))
            report['model_mae_on_baseline_rows_kw'] = float(mean_absolute_error(y[valid], pred[valid]))
        if capacity_kw: report['normalized_mae_percent_capacity'] = 100 * report['mae_kw'] / capacity_kw
        if daylight_threshold is not None:
            daylight = test['IRRADIATION'].to_numpy() > daylight_threshold
            report['daylight_rows'] = int(daylight.sum())
            if daylight.any(): report['daylight_mae_kw'] = float(mean_absolute_error(y[daylight], pred[daylight]))
        reports.append(report)
    return reports,model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True); parser.add_argument('--weather', required=True)
    parser.add_argument('--expected-inverters', type=int, default=None)
    parser.add_argument('--capacity-kw', type=float, default=None)
    parser.add_argument('--daylight-threshold', type=float, default=None, help='Irradiation threshold in verified training data units')
    parser.add_argument('--output-model', default='solar_ai_model_retrained.pkl')
    parser.add_argument('--output-report', default='evaluation_report.json')
    args=parser.parse_args()
    try:
        df=load_data(args.generation,args.weather,args.expected_inverters)
        reports,model=evaluate(df,args.capacity_kw,args.daylight_threshold)
        for r in reports: print(json.dumps(r,indent=2))
        joblib.dump(model,args.output_model)
        Path(args.output_report).write_text(json.dumps({'warning':'Retrospective sensor-input evaluation, not a 48-hour forecast test. Final saved model is the third fold model, trained on the first 80% only.', 'sklearn_version':sklearn.__version__, 'features':FEATURES, 'folds':reports},indent=2),encoding='utf-8')
        print(f'Saved model to {args.output_model}; report to {args.output_report}.')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')

if __name__ == '__main__': main()
