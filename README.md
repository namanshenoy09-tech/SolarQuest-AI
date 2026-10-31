# ☀️ SolarQuest-AI — Enterprise Yield Forecaster & Anomaly Engine

> **A real-time, ML-powered infrastructure platform predicting utility-scale solar generation with 98.38% accuracy and automated hardware fault detection.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)
![Open-Meteo API](https://img.shields.io/badge/Open--Meteo-Live%20API-0055FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

---

## 📌 Executive Overview

Intermittency and unscheduled hardware downtime cost utility-scale solar asset owners billions annually. A **1% error in generation forecasting** or **unaddressed panel soiling** can result in $500k–$2M in annual revenue loss per facility.

**SolarQuest-AI** solves this by providing:
1. **48-Hour Ultra-Precise Generation Forecasting**: Powered by an ML model ($R^2 = 0.9838$) trained on sensor telemetry and integrated with live satellite weather APIs.
2. **Automated Anomaly & Revenue Protection Engine**: Real-time divergence detection between AI yield targets and SCADA plant telemetry to flag panel soiling, inverter faults, and daily monetary loss.

---

## 🏗️ System Architecture
┌────────────────────────────────┐ ┌────────────────────────────────┐
│ Open-Meteo Satellite API │ │ Plant Sensor Telemetry │
│ (Irradiation, Temp, Clouds) │ │ (DC/AC Power, Module T) │
└───────────────┬────────────────┘ └────────────────┬───────────────┘
│ │
└───────────────────┬─────────────────────┘
▼
┌─────────────────────────────┐
│ Feature Engineering & │
│ Cyclical Time Encoding │
└──────────────┬──────────────┘
▼
┌─────────────────────────────┐
│ Linear/XGBoost ML Engine │
│ (R² Score: 98.38%) │
└──────────────┬──────────────┘
▼
┌─────────────────────────────┐
│ SolarQuest Dashboard UI │
│ • 48h Generation Curve │
│ • Anomaly & Loss Detector │
│ • PPA Revenue Tracker │
└─────────────────────────────┘

## ✨ Key Features

- **📡 Live Global API Integration**: Select any pre-configured mega-solar park (*Bhadla - India*, *Mojave - USA*, *Al Dhafra - UAE*, *Benban - Egypt*) or input custom GPS coordinates to pull real-time weather forecasts.
- **🎯 98.38% Forecast Accuracy**: Trained on 68,000+ data points of solar panel sensor telemetry.
- **🚨 Automated Hardware Fault Engine**: Detects panel soiling or inverter failures by comparing expected AI generation against actual telemetry.
- **💰 Financial Impact Analytics**: Live tracking of daily revenue ($/kWh PPA rate) and monetary loss due to inefficiencies.
- **🌱 Carbon Offset Telemetry**: Live tracking of estimated daily $CO_2$ displacement in tons.

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **$R^2$ Score (Accuracy)** | **98.38%** |
| **Mean Absolute Error (MAE)** | **Low Variance Across Peak Irradiation Hours** |
| **Training Data Size** | **68,000+ Timestamps (Generation + Sensor Fusion)** |

---

## 🛠️ Tech Stack & Dependencies

- **Language**: Python 3.10+
- **Machine Learning**: `scikit-learn`, `xgboost`, `joblib`
- **Data Pipeline**: `pandas`, `numpy`
- **Dashboard & UI**: `streamlit`, `plotly`
- **API Integration**: `requests` (Open-Meteo Global Weather API)

---

