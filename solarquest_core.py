"""Shared prototype logic. Forecasts and deviations are NOT validated operational advice."""
from __future__ import annotations
import numpy as np
import pandas as pd
import requests

FEATURES = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'HOUR']
WEATHER_URL = 'https://api.open-meteo.com/v1/forecast'


def fetch_weather(latitude: float, longitude: float) -> dict:
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError('Invalid coordinates.')
    response = requests.get(WEATHER_URL, params={
        'latitude': latitude, 'longitude': longitude,
        'hourly': 'temperature_2m,shortwave_radiation,cloud_cover',
        'forecast_days': 3, 'timezone': 'UTC',
    }, timeout=(5, 20))
    response.raise_for_status()
    return response.json()


def prepare_forecast(payload: dict, now=None, hours: int = 48) -> pd.DataFrame:
    """Select the next full UTC hours, not the first calendar-day entries."""
    if not 1 <= hours <= 48:
        raise ValueError('hours must be between 1 and 48')
    try:
        hourly = payload['hourly']
        frame = pd.DataFrame({
            'time': pd.to_datetime(hourly['time'], utc=True, errors='raise'),
            'temperature': hourly['temperature_2m'],
            'shortwave_radiation': hourly['shortwave_radiation'],
            'cloud_cover': hourly['cloud_cover'],
        })
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('Weather response is missing required hourly fields.') from exc
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    now = now.tz_convert('UTC')
    start = now.ceil('h')
    if start == now:  # exclude the already-starting hour
        start += pd.Timedelta(hours=1)
    wanted = pd.date_range(start, periods=hours, freq='h')
    if frame['time'].duplicated().any():
        raise ValueError('Weather response contains duplicate timestamps.')
    frame = frame.set_index('time').reindex(wanted)
    for col in ('temperature', 'shortwave_radiation', 'cloud_cover'):
        frame[col] = pd.to_numeric(frame[col], errors='coerce')
    if not np.isfinite(frame[['temperature','shortwave_radiation','cloud_cover']].to_numpy()).all():
        raise ValueError('Weather response is incomplete for the next full hours.')
    if (frame['shortwave_radiation'] < 0).any() or not frame['cloud_cover'].between(0, 100).all():
        raise ValueError('Weather response has invalid radiation or cloud cover values.')
    frame.index.name = 'time'
    return frame.reset_index()


def model_forecast(model, weather: pd.DataFrame, local_timezone: str) -> pd.DataFrame:
    """Training-hour feature uses plant local time, not UTC; uses assumed module temperature."""
    out = weather.copy()
    out['IRRADIATION'] = out['shortwave_radiation'] / 1000.0
    out['AMBIENT_TEMPERATURE'] = out['temperature']
    out['MODULE_TEMPERATURE'] = out['temperature'] + out['IRRADIATION'] * 25.0
    try:
        out['HOUR'] = out['time'].dt.tz_convert(local_timezone).dt.hour
    except (ValueError, KeyError) as exc:
        raise ValueError('Invalid IANA timezone, e.g. Asia/Kolkata.') from exc
    values = np.asarray(model.predict(out[FEATURES]), dtype=float)
    if len(values) != len(out) or not np.isfinite(values).all():
        raise ValueError('Model returned invalid predictions.')
    out['predicted_power_kw'] = np.where(
    out['shortwave_radiation'] <= 0,
    0.0,
    np.maximum(0.0, values)
)
    return out


def telemetry_summary(data: pd.DataFrame, ppa_rate: float, now=None):
    """Each row: completed one-hour interval, average power in kW and timezone-aware start."""
    columns = ['timestamp', 'actual_power_kw', 'expected_power_kw']
    missing = [x for x in columns if x not in data.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    if not np.isfinite(ppa_rate) or ppa_rate < 0:
        raise ValueError('PPA rate must be finite and non-negative.')
    if data.empty:
        raise ValueError('CSV is empty.')
    out = data[columns].copy()
    try:
        parsed = [pd.Timestamp(x) for x in out['timestamp']]
        if any(pd.isna(x) or x.tzinfo is None for x in parsed):
            raise ValueError('Every timestamp must include a UTC offset or Z.')
        out['timestamp'] = pd.to_datetime(parsed, utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError('Timestamps must be valid and include a UTC offset or Z.') from exc
    for col in columns[1:]:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    if not np.isfinite(out[columns[1:]].to_numpy()).all() or (out[columns[1:]] < 0).any().any():
        raise ValueError('Power values must be finite, non-negative numbers.')
    if out['timestamp'].duplicated().any():
        raise ValueError('Duplicate timestamps are not allowed.')
    if (out['timestamp'] != out['timestamp'].dt.floor('h')).any():
        raise ValueError('Intervals must begin on an exact UTC hour.')
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must include a timezone')
    if (out['timestamp'] + pd.Timedelta(hours=1) > now.tz_convert('UTC')).any():
        raise ValueError('All intervals must be completed, not future or in progress.')
    out = out.sort_values('timestamp').reset_index(drop=True)
    out['shortfall_kw'] = (out['expected_power_kw'] - out['actual_power_kw']).clip(lower=0)
    # kW average over one hour × 1 hour = kWh. No filling or extrapolating missing hours.
    shortfall_kwh = float(out['shortfall_kw'].sum())
    span_hours = int((out['timestamp'].iloc[-1] - out['timestamp'].iloc[0]) / pd.Timedelta(hours=1)) + 1
    return out, {'observed_hours': len(out), 'missing_hours': span_hours - len(out),
                 'shortfall_kwh': shortfall_kwh, 'illustrative_value': shortfall_kwh * ppa_rate}
