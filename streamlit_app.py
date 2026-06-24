import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    layout="wide"
)

#  Load model 
@st.cache_resource
def load_model():
    model = joblib.load(r"D:\fraud-detection\notebooks\models\fraud_model_reduced.pkl")
    scaler = joblib.load(r"D:\fraud-detection\notebooks\models\scaler.pkl")
    return model, scaler

model, scaler = load_model()

# title
st.title(" Credit Card Fraud Detection System")
st.markdown("Enter transaction details below to check if it's fraudulent.")
st.markdown("---")

# Sidebar 
st.sidebar.title("About")
st.sidebar.info("""
This app uses a XGBoost model trained on 284,807 
real credit card transactions to detect fraud in real time.

**Model:** XGBoost Classifier  
**Dataset:** Kaggle Credit Card Fraud  
**Key Metric:** PR-AUC  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by Tanmay")

# Input Section 
st.subheader("Transaction Input")

top_features = [
    'V14',
    'V4',
    'V12',
    'V18',
    'V8',
    'V11',
    'V1',
    'V10',
    'Time_scaled',
    'V22',
]

feature_values = {}

col1, col2 = st.columns(2)

for idx, feature in enumerate(top_features):

    with col1 if idx < 8 else col2:

        feature_values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.4f"
        )

final_features = [
    feature_values[feature]
    for feature in top_features
]


# Predict Button 
st.markdown("---")
if st.button(" Analyse Transaction", use_container_width=True):

    features_array = np.array(final_features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.error(f"""
        ##  FRAUD DETECTED
        **Fraud Probability: {probability*100:.1f}%**
        
        This transaction has been flagged as potentially fraudulent.
        """)
    else:
        st.success(f"""
        ##  TRANSACTION LEGITIMATE
        **Fraud Probability: {probability*100:.1f}%**
        
        This transaction appears to be safe.
        """)

    # Probability Gauge
    st.markdown("---")
    st.subheader("Fraud Risk Level")
    st.progress(float(probability))
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fraud Probability", f"{probability*100:.1f}%")
    col_b.metric("Legitimate Probability", f"{(1-probability)*100:.1f}%")
    col_c.metric("Verdict", "FRAUD " if prediction == 1 else "SAFE ")

    # ── SHAP Explanation ──────────────────────────────────
    st.markdown("---")
    st.subheader(" Why did the model decide this?")
    st.markdown("SHAP values show which features contributed most to this prediction.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)

        fig, ax = plt.subplots(figsize=(10, 4))
        feature_names = top_features
        
        shap.summary_plot(
            shap_values,
            features_array,
            feature_names=feature_names,
            plot_type="bar",
            show=False
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")