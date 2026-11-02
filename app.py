"""SolarQuest research prototype. Run: streamlit run app.py"""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import requests
from solarquest_core import FEATURES, fetch_weather, prepare_forecast, model_forecast, telemetry_summary

st.set_page_config(page_title='SolarQuest-AI | Research Prototype', layout='wide')
st.title('☀️ SolarQuest-AI')
st.caption('Research prototype · Forecasting and underperformance exploration')
st.warning('Not production-validated. No live SCADA connection or automatic fault diagnosis. Do not use for trading, dispatch, maintenance or settlement decisions.')

mode = st.sidebar.radio('Mode', ['Manual simulation', 'Weather forecast experiment', 'Upload hourly telemetry'])
model_path = Path('solar_ai_model.pkl')

@st.cache_resource
def load_model(path: str, modified: float):
    # NEVER load an untrusted pickle: deserialization can execute arbitrary code.
    return joblib.load(path)

def get_model():
    if not model_path.exists():
        st.error('Missing solar_ai_model.pkl. Copy your trusted model into this folder or run the training script.')
        st.stop()
    try:
        return load_model(str(model_path), model_path.stat().st_mtime)
    except Exception as exc:
        st.error(f'Could not load the model: {exc}. Check Python and scikit-learn compatibility.')
        st.stop()

if mode == 'Manual simulation':
    st.subheader('Manual weather scenario')
    st.info('Illustrative model output for the training plant; this is not a live reading or validated forecast. Confirm input units with the dataset documentation.')
    a, b = st.columns(2)
    irradiation = a.number_input('Irradiation (training-data units; confirm before use)', min_value=0.0, max_value=2.0, value=0.8, step=0.05)
    ambient = a.number_input('Ambient temperature (°C)', value=28.0)
    module = b.number_input('Module temperature (°C)', value=35.0)
    hour = b.slider('Plant-local hour', 0, 23, 12)
    model = get_model()
    x = pd.DataFrame([[irradiation, ambient, module, hour]], columns=FEATURES)
    try:
        prediction = max(0.0, float(model.predict(x)[0]))
        st.metric('Estimated power (kW, if training target is kW)', f'{prediction:,.1f}')
    except Exception as exc:
        st.error(f'Prediction failed: {exc}')

elif mode == 'Weather forecast experiment':
    st.subheader('Next 48 full hours · UTC')
    st.warning('Research experiment only. Use the training plant’s coordinates and time zone. Other plants and capacity scaling are not validated. Forecasted radiation is divided by 1,000 and module temperature is approximated, so feature units and model performance must be independently checked.')
    a,b,c = st.columns(3)
    lat = a.number_input('Training-plant latitude', min_value=-90.0, max_value=90.0, value=27.53, format='%.4f')
    lon = b.number_input('Training-plant longitude', min_value=-180.0, max_value=180.0, value=71.91, format='%.4f')
    timezone = c.text_input('Plant time zone (IANA)', value='Asia/Kolkata')
    st.caption('Default coordinates are placeholders, not a verified training-plant location. Replace them before testing.')
    if st.button('Run weather experiment'):
        model = get_model()
        try:
            with st.spinner('Requesting forecast from Open-Meteo...'):
                forecast = model_forecast(model, prepare_forecast(fetch_weather(lat, lon)), timezone)
            st.line_chart(forecast.set_index('time')[['predicted_power_kw']])
            st.dataframe(forecast[['time','shortwave_radiation','temperature','cloud_cover','predicted_power_kw']], use_container_width=True)
            st.caption('Timestamps are UTC. Values are experimental model predictions, not measured power. No energy or revenue claim is made.')
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            st.error(f'Unable to generate the experiment: {exc}')

else:
    st.subheader('Historical hourly telemetry review')
    st.info('Upload your own completed hourly data. This app does not connect to SCADA. Expected values must be independently supplied; the comparison does not diagnose a cause.')
    st.code('timestamp,actual_power_kw,expected_power_kw\n2025-01-01T10:00:00Z,75,100\n2025-01-01T11:00:00Z,80,105')
    st.caption('Each row must represent a completed one-hour interval starting at timestamp, with average kW over that interval. Include Z or a UTC offset. Do not upload data you lack permission to use.')
    rate = st.number_input('Illustrative energy price ($/kWh)', min_value=0.0, value=0.06, step=0.005, format='%.3f')
    upload = st.file_uploader('Upload CSV', type=['csv'])
    if upload:
        try:
            frame, stats = telemetry_summary(pd.read_csv(upload), rate)
            a,b,c,d=st.columns(4)
            a.metric('Observed intervals', stats['observed_hours'])
            b.metric('Missing hours in range', stats['missing_hours'])
            c.metric('Positive shortfall (kWh)', f"{stats['shortfall_kwh']:,.1f}")
            d.metric('Illustrative value', f"${stats['illustrative_value']:,.2f}")
            st.line_chart(frame.set_index('timestamp')[['actual_power_kw','expected_power_kw']])
            st.dataframe(frame, use_container_width=True)
            st.caption('Positive hourly shortfalls are summed without offsetting surpluses. The value is a scenario estimate, not verified lost revenue or recovered savings; curtailment, contracts, meter quality and other causes are not modeled. Missing hours are not extrapolated.')
        except (ValueError, pd.errors.ParserError) as exc:
            st.error(f'Unable to read telemetry: {exc}')
