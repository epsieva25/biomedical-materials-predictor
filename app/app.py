import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="Biomedical Material Property Predictor", layout="centered")

# ---------------- Load Models and Scaler with Absolute Paths ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

svr_model_path = os.path.join(BASE_DIR, "models", "svr_model.pkl")
dtr_model_path = os.path.join(BASE_DIR, "models", "dtr_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
data_path = os.path.join(BASE_DIR, "data", "sustainable_materials.csv")

try:
    svr_model = joblib.load(svr_model_path)
    dtr_model = joblib.load(dtr_model_path)
    scaler = joblib.load(scaler_path)
    df = pd.read_csv(data_path)
except Exception as e:
    st.error(f"Error loading models or dataset: {e}")
    st.stop()

# ---------------------- Streamlit UI ----------------------
st.title("🔬 Biomedical Material Property Predictor")
st.markdown("""
Predict **Bio-compatibility Score** of sustainable materials for biomedical applications using AI models (SVR & Decision Tree).
""")

# Input form
with st.form("input_form"):
    st.subheader("Enter Material Properties")
    biodegradable_index = st.slider("Biodegradable Index (0.2 - 0.9)", 0.2, 0.9, 0.5)
    tensile_strength = st.slider("Tensile Strength (30 - 120 MPa)", 30.0, 120.0, 75.0)
    porosity = st.slider("Porosity (0.1 - 0.6)", 0.1, 0.6, 0.3)
    water_absorption = st.slider("Water Absorption (1 - 10%)", 1.0, 10.0, 5.0)
    
    submitted = st.form_submit_button("Predict Score")

if submitted:
    input_data = np.array([[biodegradable_index, tensile_strength, porosity, water_absorption]])
    input_scaled = scaler.transform(input_data)

    svr_pred = svr_model.predict(input_scaled)[0]
    dtr_pred = dtr_model.predict(input_scaled)[0]

    st.success(f"✅ Predicted Bio-compatibility Score (SVR): `{svr_pred:.2f}`")
    st.success(f"✅ Predicted Bio-compatibility Score (Decision Tree): `{dtr_pred:.2f}`")

    # Add novelty: comparative recommendation
    avg_score = (svr_pred + dtr_pred) / 2
    if avg_score >= 70:
        st.markdown("🟢 **Excellent candidate for biomedical implant usage.**")
    elif avg_score >= 50:
        st.markdown("🟡 **Moderate compatibility – further testing recommended.**")
    else:
        st.markdown("🔴 **Low compatibility – consider material refinement.**")

# ---------------------- Data Explorer (Optional) ----------------------
with st.expander("📊 Show Sample Dataset"):
    st.dataframe(df.head(10))
