import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# ── Load model ───────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load(r"D:\fraud-detection\notebooks\models\fraud_model.pkl")
    scaler = joblib.load(r"D:\fraud-detection\notebooks\models\scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Title ─────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection System")
st.markdown("Enter transaction details below to check if it's fraudulent.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
This app uses a Random Forest model trained on 284,807 
real credit card transactions to detect fraud in real time.

**Model:** Random Forest  
**Dataset:** Kaggle Credit Card Fraud  
**Key Metric:** PR-AUC  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by [Your Name]")

# ── Input Section ─────────────────────────────────────────
st.subheader("Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=0.0,
        max_value=100000.0,
        value=150.0,
        step=10.0
    )
    time = st.number_input(
        "Time (seconds since first transaction)",
        min_value=0.0,
        value=50000.0
    )

with col2:
    st.markdown("**V1 – V10 Features** (PCA transformed)")
    v_features_1 = []
    for i in range(1, 11):
        val = st.number_input(f"V{i}", value=0.0, 
                              format="%.4f", key=f"v{i}")
        v_features_1.append(val)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**V11 – V20 Features**")
    v_features_2 = []
    for i in range(11, 21):
        val = st.number_input(f"V{i}", value=0.0,
                              format="%.4f", key=f"v{i}")
        v_features_2.append(val)

with col4:
    st.markdown("**V21 – V28 Features**")
    v_features_3 = []
    for i in range(21, 29):
        val = st.number_input(f"V{i}", value=0.0,
                              format="%.4f", key=f"v{i}")
        v_features_3.append(val)

# ── Scale Amount and Time ─────────────────────────────────
from sklearn.preprocessing import StandardScaler
temp_scaler = StandardScaler()

# Combine all features in correct order
all_features = v_features_1 + v_features_2 + v_features_3

# Scale amount and time manually using same logic as training
amount_scaled = (amount - 88.35) / 250.12
time_scaled = (time - 94813.86) / 47488.15

# Final feature vector: V1-V28 + Amount_scaled + Time_scaled
final_features = all_features + [amount_scaled, time_scaled]

# ── Predict Button ────────────────────────────────────────
st.markdown("---")
if st.button("🔍 Analyse Transaction", use_container_width=True):

    features_array = np.array(final_features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.error(f"""
        ## 🚨 FRAUD DETECTED
        **Fraud Probability: {probability*100:.1f}%**
        
        This transaction has been flagged as potentially fraudulent.
        """)
    else:
        st.success(f"""
        ## ✅ TRANSACTION LEGITIMATE
        **Fraud Probability: {probability*100:.1f}%**
        
        This transaction appears to be safe.
        """)

    # ── Probability Gauge ─────────────────────────────────
    st.markdown("---")
    st.subheader("Fraud Risk Level")
    st.progress(float(probability))
    
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fraud Probability", f"{probability*100:.1f}%")
    col_b.metric("Legitimate Probability", f"{(1-probability)*100:.1f}%")
    col_c.metric("Verdict", "FRAUD 🚨" if prediction == 1 else "SAFE ✅")

    # ── SHAP Explanation ──────────────────────────────────
    st.markdown("---")
    st.subheader("🔎 Why did the model decide this?")
    st.markdown("SHAP values show which features contributed most to this prediction.")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_array)

        fig, ax = plt.subplots(figsize=(10, 4))
        feature_names = [f"V{i}" for i in range(1, 29)] + \
                        ["Amount_scaled", "Time_scaled"]
        
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