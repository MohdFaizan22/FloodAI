# 🌊 FloodAI — Urban Flood Risk Assessment & Early Warning System

> **⚠️ IMPORTANT DISCLAIMER:** This project uses a **synthetic** dataset generated for educational and research purposes. It is **NOT** an official government flood-warning or disaster-management system. All predictions and risk scores are demonstrations only and must not be used for real emergency decisions.

---

## 📋 Project Overview

FloodAI is a complete end-to-end machine learning project that demonstrates how AI and data science can support flood risk assessment in urban environments. The system ingests environmental and geographical features, trains multiple ML models, generates a probabilistic flood-risk score, and delivers insights through an interactive Streamlit dashboard.

**Aligned with:**
| SDG | Goal |
|-----|------|
| 🌊 SDG 6 | Clean Water and Sanitation |
| 🏙️ SDG 11 | Sustainable Cities and Communities |
| 🌍 SDG 13 | Climate Action |

---

## 🎯 Problem Statement

Urban flooding is one of the most frequent and costly natural disasters globally. Early identification of high-risk areas requires combining multiple environmental signals — rainfall intensity, river discharge, water levels, elevation, soil permeability, land cover, and population density — into an actionable risk score.

This project demonstrates how machine learning can:
- Combine multi-dimensional environmental features into a flood risk probability
- Identify the most influential predictors of flood occurrence
- Provide explainable, interpretable predictions via SHAP
- Visualise geographic risk patterns on an interactive map

---

## 📊 Dataset Description

| Property | Value |
|----------|-------|
| Source | Synthetic dataset (locally stored) |
| Rows | 10,000 |
| Columns | 14 |
| Target | `Flood Occurred` (binary: 0 = No Flood, 1 = Flood) |

> ⚠️ **The dataset is entirely synthetic.** It does not represent real historical flood measurements or official meteorological records. Do not treat model outputs as real-world flood forecasts.

### Features

| Column | Type | Description |
|--------|------|-------------|
| Latitude | Numerical | Geographic latitude of the observation |
| Longitude | Numerical | Geographic longitude of the observation |
| Rainfall (mm) | Numerical | Recorded or estimated rainfall |
| Temperature (°C) | Numerical | Ambient temperature |
| Humidity (%) | Numerical | Relative humidity |
| River Discharge (m³/s) | Numerical | River flow rate |
| Water Level (m) | Numerical | Current water body level |
| Elevation (m) | Numerical | Terrain elevation above sea level |
| Land Cover | Categorical | Land use type (Urban, Forest, Agricultural, Desert, Water Body) |
| Soil Type | Categorical | Soil classification (Clay, Loam, Sandy, Silt, Peat) |
| Population Density | Numerical | People per unit area |
| Infrastructure | Binary | 0 = Poor infrastructure, 1 = Good infrastructure |
| Historical Floods | Binary | 0 = No prior flood record, 1 = Prior flood record |
| Flood Occurred | Binary (Target) | 0 = No Flood, 1 = Flood |

---

## 🔧 Methodology

### 1. Data Preprocessing
- Duplicate record removal
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling for numerical features
- One-Hot Encoding for categorical features
- Conservative 3×IQR outlier clipping
- Strict train/test split (80/20, stratified) to prevent data leakage

### 2. Feature Engineering
| Engineered Feature | Description |
|-------------------|-------------|
| Rainfall_WaterLevel_Interaction | Normalised rainfall × water level |
| Rainfall_Discharge_Interaction | Normalised rainfall × river discharge |
| Rainfall_Intensity | Rainfall / (1 + Humidity/100) |
| WaterLevel_Severity | Water level weighted by elevation |
| Discharge_Risk | Log1p of river discharge |
| Low_Elevation_Flag | Binary: 1 if elevation < 500 m |
| Population_Exposure | Population density in low-elevation areas |

### 3. Machine Learning Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline linear model with balanced class weights |
| Random Forest | 200 trees, max_depth=12, class_weight="balanced" |
| XGBoost | 200 estimators, scale_pos_weight to handle class imbalance |

### 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### 5. Explainable AI (SHAP)
- **Global explanations:** Mean |SHAP value| bar chart, beeswarm plot
- **Individual explanations:** Per-prediction SHAP waterfall / bar chart
- SHAP values describe model associations, not physical causation.

### 6. Risk Scoring

> ⚠️ These are **application-defined visualisation categories**, not official government flood-warning thresholds.

| Probability Range | Risk Category |
|------------------|--------------|
| 0% – 25% | 🟢 Low Risk |
| 25% – 50% | 🟡 Moderate Risk |
| 50% – 75% | 🟠 High Risk |
| 75% – 100% | 🔴 Very High Risk |

---

## 🖥️ Interactive Dashboard

The Streamlit application provides 6 sections:

1. **📊 Dashboard** — KPI metrics, flood distribution, rainfall trends
2. **🔬 Data Analysis** — Feature distributions, correlation matrix, geographic scatter
3. **🎯 Flood Prediction** — Interactive form for real-time flood risk assessment
4. **🗺️ Risk Map** — Interactive Plotly map with risk category filter
5. **📈 Model Performance** — Comparison table, confusion matrices, ROC/PR curves
6. **🧠 Explainable AI** — SHAP global + individual prediction explanations

---

## 🗂️ Project Architecture

```
FloodAI/
├── .venv/                        # Local Python virtual environment
├── data/
│   ├── raw/
│   │   └── flood_risk_dataset_india.csv   # Original synthetic dataset
│   └── processed/
│       ├── flood_risk_processed.csv
│       ├── flood_risk_engineered.csv
│       └── test_predictions.csv
├── models/
│   ├── best_model_pipeline.pkl   # Best trained model (sklearn Pipeline)
│   ├── model_meta.json           # Feature names & metadata
│   ├── Logistic_Regression_pipeline.pkl
│   ├── Random_Forest_pipeline.pkl
│   └── XGBoost_pipeline.pkl
├── outputs/
│   ├── figures/                  # EDA, SHAP, ROC, CM charts
│   └── reports/
│       ├── dataset_validation.json
│       └── model_metrics.json
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data loading, cleaning, ColumnTransformer
│   ├── feature_engineering.py    # Engineered features
│   ├── eda.py                    # EDA visualisations
│   ├── train.py                  # Model training & evaluation
│   ├── prediction.py             # Scoring utilities
│   └── explainability.py         # SHAP explanations
├── app/
│   └── app.py                    # Streamlit dashboard
├── notebooks/
│   └── FloodAI_Analysis.ipynb    # Complete analysis notebook
├── requirements.txt
├── README.md
├── .gitignore
└── run_training.py               # Single-command training pipeline
```

---

## 🚀 Local Installation

### Prerequisites
- Python 3.10 or 3.11
- Git (optional)

### Setup

```powershell
# 1. Navigate into the project folder
cd FloodAI

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
python -m pip install -r requirements.txt

# 6. Run the complete training pipeline
python run_training.py

# 7. Launch the Streamlit application
streamlit run app/app.py
```

The application will be available at `http://localhost:8501`.

---

## 📦 Deployment (Free — Streamlit Community Cloud)

1. Push this repository to GitHub (exclude `.venv/` and `models/*.pkl`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set **Main file path** to `app/app.py`
5. Deploy

> **Note:** For cloud deployment, add a setup step to run `python run_training.py` before the app starts, or commit the trained model files to the repository (if size permits).

### Hugging Face Spaces

1. Create a new Space (Streamlit SDK)
2. Push project files
3. Add a `packages.txt` if system dependencies are needed
4. The Space will auto-install `requirements.txt` and run `app/app.py`

---

## ⚠️ Limitations

- Dataset is **entirely synthetic** — results do not reflect real flood patterns
- Models are trained on simulated data and may not generalise to real-world conditions
- Geographic coordinates are randomly distributed and do not represent actual flood-prone areas
- Risk categories are arbitrary visualisation thresholds
- The system has not been validated against official meteorological or hydrological records
- Do not use for real emergency or disaster management decisions

---

## 🔭 Future Improvements

- Integrate real historical flood datasets (IMD, NDMA, Copernicus)
- Add time-series modelling for flood forecasting
- Incorporate satellite imagery features
- Deploy on government open-data infrastructure
- Add multi-language support for Indian regional languages
- Real-time data ingestion from weather APIs
- Integration with early-warning notification systems (SMS/email)
- Uncertainty quantification using Bayesian methods

---

## 📄 License

MIT License — for educational and research use only.

---

*Made with ❤️ as a UG Major Project · FloodAI v1.0*
