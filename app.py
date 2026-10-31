import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(page_title="SolarCast AI ⚡", layout="centered")

st.title("☀️ SolarCast — AI Power Forecaster")
st.write("Predict solar plant power output in real-time using Machine Learning.")

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('solar_ai_model.pkl')

model = load_model()

st.sidebar.header("🎛️ Weather Controls")

# Sliders for user input
irradiation = st.sidebar.slider("Sunlight Intensity (Irradiation W/m²)", 0.0, 1.5, 0.8, 0.05)
ambient_temp = st.sidebar.slider("Ambient Temperature (°C)", 10.0, 50.0, 28.0, 0.5)
module_temp = st.sidebar.slider("Panel Module Temp (°C)", 10.0, 70.0, 35.0, 0.5)
hour = st.sidebar.slider("Hour of Day", 0, 23, 12, 1)

# Predict button
input_data = pd.DataFrame({
    'IRRADIATION': [irradiation],
    'AMBIENT_TEMPERATURE': [ambient_temp],
    'MODULE_TEMPERATURE': [module_temp],
    'HOUR': [hour]
})

prediction = model.predict(input_data)[0]
# Prevent negative power predictions at night
prediction = max(0.0, prediction)

st.markdown("---")
st.subheader("📊 Forecast Results")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Predicted AC Power", value=f"{prediction:,.1f} kW")

with col2:
    if prediction > 15000:
        st.success("🟢 High Output (Peak Hours)")
    elif prediction > 2000:
        st.warning("🟡 Moderate Output")
    else:
        st.error("🔴 Low / Zero Output (Night/Clouds)")

# Simple daily forecast curve simulation
hours = np.arange(0, 24)
simulated_irrad = np.maximum(0, np.sin((hours - 6) * np.pi / 12) * irradiation)
simulated_inputs = pd.DataFrame({
    'IRRADIATION': simulated_irrad,
    'AMBIENT_TEMPERATURE': [ambient_temp]*24,
    'MODULE_TEMPERATURE': [module_temp]*24,
    'HOUR': hours
})
daily_preds = np.maximum(0, model.predict(simulated_inputs))

chart_data = pd.DataFrame({"Hour": hours, "Predicted Power (kW)": daily_preds})
st.subheader("📈 Simulated 24-Hour Output Curve")
st.line_chart(chart_data.set_index("Hour"))
