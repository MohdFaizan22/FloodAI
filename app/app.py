"""
app/app.py
----------
FloodAI — Urban Flood Risk Assessment & Early Warning System
Interactive Streamlit Dashboard

DISCLAIMER: This application uses a SYNTHETIC dataset and is a
student/research project. It is NOT an official government disaster
warning system. Predictions are for educational and demonstration
purposes only. Risk categories are application-defined visualisation
thresholds, not official government flood-warning standards.

Run:
    streamlit run app/app.py
"""

import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── project root ──────────────────────────────────────────────────────────────
APP_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_RAW_PATH   = os.path.join(PROJECT_ROOT, "data", "raw",       "flood_risk_dataset_india.csv")
DATA_PROC_PATH  = os.path.join(PROJECT_ROOT, "data", "processed", "flood_risk_engineered.csv")
PRED_PATH       = os.path.join(PROJECT_ROOT, "data", "processed", "test_predictions.csv")
MODEL_PATH      = os.path.join(PROJECT_ROOT, "models", "best_model_pipeline.pkl")
META_PATH       = os.path.join(PROJECT_ROOT, "models", "model_meta.json")
METRICS_PATH    = os.path.join(PROJECT_ROOT, "outputs", "reports", "model_metrics.json")
FIGURES_DIR     = os.path.join(PROJECT_ROOT, "outputs", "figures")

logging.basicConfig(level=logging.WARNING)

# ── Streamlit page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodAI — Urban Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .risk-low       { color: #16a34a; font-weight: 700; }
    .risk-moderate  { color: #d97706; font-weight: 700; }
    .risk-high      { color: #dc2626; font-weight: 700; }
    .risk-veryhigh  { color: #7f1d1d; font-weight: 700; }
    .disclaimer {
        background: #fef9c3;
        border: 1px solid #fde68a;
        border-radius: 6px;
        padding: 10px 14px;
        font-size: 13px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cached loaders
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner="Loading dataset …")
def load_data():
    if os.path.exists(DATA_PROC_PATH):
        return pd.read_csv(DATA_PROC_PATH)
    return pd.read_csv(DATA_RAW_PATH)


@st.cache_data
def load_predictions():
    if os.path.exists(PRED_PATH):
        return pd.read_csv(PRED_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ── risk helpers ──────────────────────────────────────────────────────────────
RISK_COLORS = {
    "Low": "#16a34a",
    "Moderate": "#d97706",
    "High": "#dc2626",
    "Very High": "#7f1d1d",
}

def prob_to_category(p):
    if p < 0.25: return "Low"
    if p < 0.50: return "Moderate"
    if p < 0.75: return "High"
    return "Very High"


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Erioll_world.svg/480px-Erioll_world.svg.png",
             width=60)
    st.title("🌊 FloodAI")
    st.caption("Urban Flood Risk Assessment")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔬 Data Analysis", "🎯 Flood Prediction",
         "🗺️ Risk Map", "📈 Model Performance", "🧠 Explainable AI"],
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px; color:#57606a;'>
    <b>SDG Alignment</b><br>
    🌊 SDG 6 — Clean Water<br>
    🏙️ SDG 11 — Sustainable Cities<br>
    🌍 SDG 13 — Climate Action<br><br>
    <b>⚠️ Synthetic Dataset</b><br>
    For educational purposes only.
    </div>
    """, unsafe_allow_html=True)

# ── load artifacts ────────────────────────────────────────────────────────────
model   = load_model()
df      = load_data()
preds   = load_predictions()
metrics = load_metrics()
meta    = load_meta()

TARGET = "Flood Occurred"
DISCLAIMER = """
<div class='disclaimer'>
⚠️ <b>Disclaimer:</b> This application uses a <b>synthetic</b> dataset for educational/research purposes.
It is <b>NOT</b> an official government flood-warning system. Risk categories are
application-defined visualisation thresholds, not official government standards.
Do not use for real emergency decisions.
</div>
"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dashboard
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.title("📊 FloodAI — Urban Flood Risk Dashboard")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    # ── KPI metrics ───────────────────────────────────────────────────────────
    total     = len(df)
    n_flood   = int(df[TARGET].sum()) if TARGET in df.columns else "N/A"
    n_noflood = total - n_flood if isinstance(n_flood, int) else "N/A"
    avg_rain  = df["Rainfall (mm)"].mean() if "Rainfall (mm)" in df.columns else 0
    avg_wl    = df["Water Level (m)"].mean() if "Water Level (m)" in df.columns else 0

    high_risk_count = 0
    if preds is not None and "Risk_Category" in preds.columns:
        high_risk_count = int((preds["Risk_Category"].isin(["High", "Very High"])).sum())

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Records",      f"{total:,}")
    c2.metric("Flood Events",        f"{n_flood:,}")
    c3.metric("Non-Flood Events",    f"{n_noflood:,}")
    c4.metric("Avg Rainfall (mm)",   f"{avg_rain:.1f}")
    c5.metric("Avg Water Level (m)", f"{avg_wl:.2f}")
    c6.metric("High/Very High Risk", f"{high_risk_count:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Flood distribution pie
        if TARGET in df.columns:
            vc = df[TARGET].value_counts()
            fig = px.pie(
                values=vc.values,
                names=["No Flood", "Flood"],
                color_discrete_sequence=["#3b82d4", "#e74c3c"],
                title="Flood vs Non-Flood Distribution",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk category distribution
        if preds is not None and "Risk_Category" in preds.columns:
            rc = preds["Risk_Category"].value_counts().reset_index()
            rc.columns = ["Risk Category", "Count"]
            fig = px.bar(
                rc, x="Risk Category", y="Count",
                color="Risk Category",
                color_discrete_map=RISK_COLORS,
                title="Predicted Risk Category Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        # Rainfall distribution
        fig = px.histogram(
            df, x="Rainfall (mm)", color=TARGET if TARGET in df.columns else None,
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
            title="Rainfall Distribution by Flood Occurrence",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Water level distribution
        fig = px.histogram(
            df, x="Water Level (m)", color=TARGET if TARGET in df.columns else None,
            nbins=50, barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
            title="Water Level Distribution by Flood Occurrence",
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Analysis
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Data Analysis":
    st.title("🔬 Data Analysis")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Statistics", "📊 Distributions", "🔗 Correlations", "🌍 Geographic"])

    with tab1:
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isnull().sum().sum()))
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

    with tab2:
        feat = st.selectbox(
            "Select feature",
            [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET],
        )
        fig = px.histogram(
            df, x=feat,
            color=TARGET if TARGET in df.columns else None,
            nbins=60, barmode="overlay", opacity=0.7,
            color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
            title=f"{feat} Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        if "Land Cover" in df.columns:
            fig2 = px.histogram(
                df, x="Land Cover",
                color=TARGET if TARGET in df.columns else None,
                barmode="group",
                color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
                title="Flood Occurrence by Land Cover",
            )
            st.plotly_chart(fig2, use_container_width=True)

        if "Soil Type" in df.columns:
            fig3 = px.histogram(
                df, x="Soil Type",
                color=TARGET if TARGET in df.columns else None,
                barmode="group",
                color_discrete_map={0: "#3b82d4", 1: "#e74c3c"},
                title="Flood Occurrence by Soil Type",
            )
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        num_df = df.select_dtypes(include=[np.number])
        corr   = num_df.corr()
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Feature Correlation Matrix", text_auto=".2f",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rainfall vs Water Level")
        fig2 = px.scatter(
            df.sample(min(2000, len(df)), random_state=42),
            x="Rainfall (mm)", y="Water Level (m)",
            color=TARGET if TARGET in df.columns else None,
            opacity=0.5,
            color_continuous_scale="RdBu_r",
            title="Rainfall vs Water Level",
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        if "Latitude" in df.columns and "Longitude" in df.columns:
            sample = df.sample(min(3000, len(df)), random_state=42)
            fig = px.scatter_map(
                sample,
                lat="Latitude", lon="Longitude",
                color=TARGET if TARGET in df.columns else None,
                color_continuous_scale="RdYlGn_r",
                map_style="open-street-map",
                zoom=3.5, center={"lat": 22, "lon": 82},
                title="Geographic Distribution of Flood Occurrences",
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latitude/Longitude columns not found.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Flood Prediction
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Flood Prediction":
    st.title("🎯 Flood Risk Prediction")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)
    st.markdown("Enter environmental and geographical conditions to get a flood risk assessment.")

    # Build input ranges from actual data
    def _range(col):
        if col in df.columns:
            return float(df[col].min()), float(df[col].max()), float(df[col].mean())
        return 0.0, 100.0, 50.0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Rainfall & Water")
        mn, mx, mean = _range("Rainfall (mm)")
        rainfall = st.slider("Rainfall (mm)", mn, mx, mean)

        mn, mx, mean = _range("Humidity (%)")
        humidity = st.slider("Humidity (%)", mn, mx, mean)

        mn, mx, mean = _range("River Discharge (m³/s)")
        river_discharge = st.slider("River Discharge (m³/s)", mn, mx, mean)

        mn, mx, mean = _range("Water Level (m)")
        water_level = st.slider("Water Level (m)", mn, mx, mean)

    with col2:
        st.subheader("Geography & Climate")
        mn, mx, mean = _range("Latitude")
        latitude = st.slider("Latitude", mn, mx, mean)

        mn, mx, mean = _range("Longitude")
        longitude = st.slider("Longitude", mn, mx, mean)

        mn, mx, mean = _range("Elevation (m)")
        elevation = st.slider("Elevation (m)", mn, mx, mean)

        mn, mx, mean = _range("Temperature (°C)")
        temperature = st.slider("Temperature (°C)", mn, mx, mean)

    with col3:
        st.subheader("Infrastructure & History")
        mn, mx, mean = _range("Population Density")
        pop_density = st.slider("Population Density", mn, mx, mean)

        infrastructure = st.selectbox("Infrastructure (0=Poor, 1=Good)", [0, 1])
        historical_floods = st.selectbox("Historical Floods (0=No, 1=Yes)", [0, 1])

        land_cover_opts = sorted(df["Land Cover"].unique().tolist()) if "Land Cover" in df.columns else ["Urban"]
        land_cover = st.selectbox("Land Cover", land_cover_opts)

        soil_opts = sorted(df["Soil Type"].unique().tolist()) if "Soil Type" in df.columns else ["Clay"]
        soil_type = st.selectbox("Soil Type", soil_opts)

    if st.button("🔍 Assess Flood Risk", type="primary", use_container_width=True):
        input_dict = {
            "Latitude":              latitude,
            "Longitude":             longitude,
            "Rainfall (mm)":         rainfall,
            "Temperature (°C)":      temperature,
            "Humidity (%)":          humidity,
            "River Discharge (m³/s)": river_discharge,
            "Water Level (m)":       water_level,
            "Elevation (m)":         elevation,
            "Land Cover":            land_cover,
            "Soil Type":             soil_type,
            "Population Density":    pop_density,
            "Infrastructure":        infrastructure,
            "Historical Floods":     historical_floods,
        }

        # Apply feature engineering to single row
        try:
            from src.feature_engineering import engineer_features
            row_df = engineer_features(pd.DataFrame([input_dict]))
        except Exception:
            row_df = pd.DataFrame([input_dict])

        try:
            proba = float(model.predict_proba(row_df)[0, 1])
            pred  = int(model.predict(row_df)[0])
            cat   = prob_to_category(proba)

            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Flood Probability", f"{proba:.1%}")
            r2.metric("Risk Percentage",   f"{proba*100:.1f}%")
            r3.metric("Risk Category",     cat)
            r4.metric("Prediction",        "🌊 FLOOD" if pred == 1 else "✅ No Flood")

            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Flood Risk Score"},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLORS.get(cat, "#888")},
                    "steps": [
                        {"range": [0, 25],  "color": "#dcfce7"},
                        {"range": [25, 50], "color": "#fef9c3"},
                        {"range": [50, 75], "color": "#fed7aa"},
                        {"range": [75, 100],"color": "#fecaca"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": proba * 100,
                    },
                },
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # SHAP individual explanation
            st.subheader("🧠 Top Contributing Factors")
            try:
                from src.explainability import explain_single_prediction
                feat_names = meta.get("feature_names", [])
                if feat_names:
                    contribs = explain_single_prediction(model, row_df, feat_names)
                    top10 = dict(list(contribs.items())[:10])
                    fig2 = go.Figure(go.Bar(
                        x=list(top10.values()),
                        y=list(top10.keys()),
                        orientation="h",
                        marker_color=["#e74c3c" if v > 0 else "#3b82d4" for v in top10.values()],
                    ))
                    fig2.update_layout(title="SHAP Feature Contributions (this prediction)",
                                       xaxis_title="SHAP Value", height=380)
                    st.plotly_chart(fig2, use_container_width=True)
                    st.caption("Red = increases flood risk | Blue = decreases flood risk")
            except Exception as exc:
                st.info(f"SHAP explanation unavailable: {exc}")

        except Exception as exc:
            st.error(f"Prediction error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Risk Map
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ Risk Map":
    st.title("🗺️ Flood Risk Map")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    if preds is None or "Latitude" not in preds.columns:
        st.warning("Risk map data not available. Run `python run_training.py` first.")
    else:
        cat_filter = st.multiselect(
            "Filter by Risk Category",
            ["Low", "Moderate", "High", "Very High"],
            default=["Low", "Moderate", "High", "Very High"],
        )
        filtered = preds[preds["Risk_Category"].isin(cat_filter)] if "Risk_Category" in preds.columns else preds
        sample   = filtered.sample(min(3000, len(filtered)), random_state=42)

        fig = px.scatter_map(
            sample,
            lat="Latitude", lon="Longitude",
            color="Risk_Category",
            color_discrete_map=RISK_COLORS,
            hover_data={
                "Flood_Probability": ":.3f",
                "Risk_Category": True,
                "Flood Occurred": True if "Flood Occurred" in sample.columns else False,
            },
            map_style="open-street-map",
            zoom=3.5, center={"lat": 22, "lon": 82},
            title=f"Predicted Flood Risk Map ({len(sample):,} points shown)",
            height=600,
            size_max=8,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk Category Summary")
        rc = filtered["Risk_Category"].value_counts().reset_index()
        rc.columns = ["Category", "Count"]
        rc["Percentage"] = (rc["Count"] / len(filtered) * 100).round(2)
        st.dataframe(rc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    if metrics is None:
        st.warning("No metrics found. Run `python run_training.py` first.")
    else:
        st.success(f"Best Model: **{metrics.get('best_model', 'N/A')}**")

        model_rows = metrics.get("models", [])
        if model_rows:
            display_cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
            rows = [{k: round(v, 4) if isinstance(v, float) else v
                     for k, v in m.items() if k in display_cols}
                    for m in model_rows]
            st.dataframe(pd.DataFrame(rows).set_index("model"), use_container_width=True)

            # Grouped bar chart
            names    = [r["model"] for r in rows]
            mkeys    = ["accuracy", "precision", "recall", "f1", "roc_auc"]
            fig = go.Figure()
            for mk in mkeys:
                fig.add_trace(go.Bar(
                    name=mk.capitalize(),
                    x=names,
                    y=[r[mk] for r in rows],
                ))
            fig.update_layout(barmode="group", title="Model Comparison",
                              yaxis_range=[0, 1.05], height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Saved figure images
        st.subheader("Confusion Matrices")
        for mname in ["Logistic_Regression", "Random_Forest", "XGBoost"]:
            path = os.path.join(FIGURES_DIR, f"confusion_matrix_{mname}.png")
            if os.path.exists(path):
                st.image(path, caption=mname.replace("_", " "), use_container_width=False, width=400)

        col1, col2 = st.columns(2)
        with col1:
            roc_path = os.path.join(FIGURES_DIR, "roc_curves_all.png")
            if os.path.exists(roc_path):
                st.image(roc_path, caption="ROC Curves", use_container_width=True)
        with col2:
            pr_path = os.path.join(FIGURES_DIR, "pr_curves_all.png")
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curves", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Explainable AI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Explainable AI":
    st.title("🧠 Explainable AI")
    st.markdown(DISCLAIMER, unsafe_allow_html=True)

    st.info("""
    **DISCLAIMER:** Feature importances and SHAP values describe statistical associations
    learned by the model. They do NOT establish causal relationships between features and
    flood occurrence. These values describe *model behaviour*, not physical causation.
    """)

    tab1, tab2 = st.tabs(["🌍 Global Importance", "🔍 Individual Prediction"])

    with tab1:
        st.subheader("Global Feature Importance (SHAP)")
        shap_bar = os.path.join(FIGURES_DIR, "shap_importance.png")
        shap_sum = os.path.join(FIGURES_DIR, "shap_summary.png")

        if os.path.exists(shap_bar):
            st.image(shap_bar, caption="Mean |SHAP value| — Global Feature Importance",
                     use_container_width=True)
        if os.path.exists(shap_sum):
            st.image(shap_sum, caption="SHAP Summary (Beeswarm) Plot",
                     use_container_width=True)

        # Also show sklearn feature importances via prediction.py
        st.subheader("Model Feature Importance")
        try:
            from src.prediction import get_feature_importance
            feat_names = meta.get("feature_names", [])
            if feat_names:
                fi = get_feature_importance(model, feat_names)
                top20 = fi.head(20)
                fig = px.bar(top20, x="importance", y="feature",
                             orientation="h",
                             color="importance",
                             color_continuous_scale="Blues",
                             title="Top 20 Features by Importance")
                fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance unavailable: {exc}")

    with tab2:
        st.subheader("Explain an Individual Prediction")
        st.markdown("Select a test observation to see which features drove the prediction.")

        if preds is not None:
            idx = st.slider("Select test observation index", 0, len(preds) - 1, 0)
            row_data = preds.iloc[[idx]]

            # Reconstruct original features (drop prediction columns)
            pred_extra = ["Flood_Probability", "Risk_Percentage",
                          "Risk_Category", "Flood_Prediction", "Flood Occurred"]
            feature_row = row_data.drop(columns=[c for c in pred_extra if c in row_data.columns])

            st.write("**Selected observation:**")
            st.dataframe(feature_row.T.rename(columns={feature_row.index[0]: "Value"}),
                         use_container_width=True)

            if st.button("Explain This Prediction"):
                try:
                    from src.explainability import explain_single_prediction
                    feat_names = meta.get("feature_names", [])
                    if feat_names:
                        contribs = explain_single_prediction(model, feature_row, feat_names)
                        top10 = dict(list(contribs.items())[:10])
                        fig = go.Figure(go.Bar(
                            x=list(top10.values()),
                            y=list(top10.keys()),
                            orientation="h",
                            marker_color=["#e74c3c" if v > 0 else "#3b82d4"
                                          for v in top10.values()],
                        ))
                        fig.update_layout(
                            title="SHAP Feature Contributions",
                            xaxis_title="SHAP Value",
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Red bars increase flood risk prediction | Blue bars decrease it")
                    else:
                        st.warning("Feature names not available. Re-run training pipeline.")
                except Exception as exc:
                    st.error(f"Explanation error: {exc}")
        else:
            st.info("Test predictions not found. Run `python run_training.py` first.")

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:12px; color:#57606a;'>"
    "FloodAI v1.0 · Synthetic Dataset · Educational Use Only · "
    "Aligned with UN SDG 6, 11 & 13"
    "</div>",
    unsafe_allow_html=True,
)
