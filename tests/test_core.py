import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from solarquest_core import prepare_forecast, model_forecast, telemetry_summary
from train_evaluate import evaluate, load_data

def weather_payload():
    times = pd.date_range('2025-01-01', periods=72, freq='h', tz='UTC')
    return {'hourly': {'time':times.strftime('%Y-%m-%dT%H:%M').tolist(), 'temperature_2m':[20]*72, 'shortwave_radiation':[500]*72, 'cloud_cover':[10]*72}}

def test_next_48_future_hours_and_local_hour():
    data=prepare_forecast(weather_payload(), now='2025-01-01T12:30:00Z')
    assert len(data)==48 and data.time.iloc[0] == pd.Timestamp('2025-01-01T13:00Z')
    class Model:
        def predict(self, x):
            assert x.HOUR.iloc[0] == 18  # Asia/Kolkata at 13:00 UTC
            return np.full(len(x), 5)
    result=model_forecast(Model(),data,'Asia/Kolkata')
    assert result.predicted_power_kw.sum()==240

def test_incomplete_weather_rejected():
    with pytest.raises(ValueError, match='incomplete'):
        prepare_forecast(weather_payload(), now='2025-01-02T23:30Z')

def test_telemetry_shortfall_and_gaps():
    x=pd.DataFrame({'timestamp':['2025-01-01T10:00Z','2025-01-01T12:00Z'], 'actual_power_kw':[75,110], 'expected_power_kw':[100,100]})
    _, summary=telemetry_summary(x,.06,now='2025-01-02T00:00Z')
    assert summary['shortfall_kwh']==25 and summary['illustrative_value']==1.5 and summary['missing_hours']==1

def test_bad_telemetry_rejected():
    x=pd.DataFrame({'timestamp':['2025-01-01 10:00'], 'actual_power_kw':[1], 'expected_power_kw':[2]})
    with pytest.raises(ValueError, match='offset'):
        telemetry_summary(x,.06,now='2025-01-02T00:00Z')
    x['timestamp']=['2025-01-02T00:00Z']
    with pytest.raises(ValueError, match='completed'):
        telemetry_summary(x,.06,now='2025-01-02T00:30Z')

def test_expanding_temporal_folds():
    times=pd.date_range('2025-01-01', periods=120, freq='h')
    x=np.arange(120, dtype=float)
    df=pd.DataFrame({'DATE_TIME':times,'IRRADIATION':x, 'AMBIENT_TEMPERATURE':x*0, 'MODULE_TEMPERATURE':x*0, 'HOUR':times.hour, 'AC_POWER':2*x})
    report,model=evaluate(df, capacity_kw=500)
    assert len(report)==3 and report[2]['train_rows']==96
    assert all(r['train_end'] < r['test_start'] for r in report)
    assert report[-1]['baseline_matched_rows']==24
    assert report[-1]['mae_kw'] < 1e-8

def test_load_rejects_duplicate_inverter(tmp_path):
    gen=pd.DataFrame({'DATE_TIME':['01-01-2025 00:00']*2,'SOURCE_KEY':['x']*2,'AC_POWER':[1,2]})
    weather=pd.DataFrame({'DATE_TIME':['2025-01-01 00:00'],'IRRADIATION':[0],'AMBIENT_TEMPERATURE':[20],'MODULE_TEMPERATURE':[20]})
    g=tmp_path/'g.csv'; w=tmp_path/'w.csv';gen.to_csv(g,index=False);weather.to_csv(w,index=False)
    with pytest.raises(ValueError,match='Duplicate timestamp'):
        load_data(g,w)
