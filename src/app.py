# =========================================
# 🚀 JUDGE-WINNING STREAMLIT APP (FINAL FIXED)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(page_title="Pharmacy Risk AI", layout="wide")

st.title("💊 AI-Powered Medication Adherence Dashboard")
st.markdown("Predict • Explain • Monitor patient refill risk")

# =========================================
# LOAD DATA + MODELS
# =========================================
@st.cache_data
def load_data():
    df = pd.read_csv("test_data.csv")
    df["SRVC_DT"] = pd.to_datetime(df["SRVC_DT"], errors="coerce")
    return df

@st.cache_resource
def load_models():
    xgb = joblib.load("xgb_model.pkl")
    xgb_tuned = joblib.load("xgb_tuned_model.pkl")
    return xgb, xgb_tuned

df = load_data()
xgb, xgb_tuned = load_models()

# =========================================
# 🔥 FEATURE ENGINEERING (CRITICAL FIX)
# =========================================

df = df.sort_values(["DESYNPUF_ID", "PROD_SRVC_ID", "SRVC_DT"])

df["prev_SRVC_DT"] = df.groupby(["DESYNPUF_ID", "PROD_SRVC_ID"])["SRVC_DT"].shift(1)

df["gap_prev"] = (df["SRVC_DT"] - df["prev_SRVC_DT"]).dt.days

df["patient_gap_mean"] = df.groupby(["DESYNPUF_ID", "PROD_SRVC_ID"])["gap_prev"].transform("mean")
df["patient_gap_std"] = df.groupby(["DESYNPUF_ID", "PROD_SRVC_ID"])["gap_prev"].transform("std")

df["gap_var_3"] = df.groupby(["DESYNPUF_ID", "PROD_SRVC_ID"])["gap_prev"].transform(
    lambda x: x.rolling(3, min_periods=1).var()
)

df["early_refill"] = (df["gap_prev"] < 0).astype(int)

df["adherence_score"] = df["DAYS_SUPLY_NUM"] / (df["gap_prev"] + 1)

df["drug_class_freq"] = df.groupby("PROD_SRVC_ID")["DESYNPUF_ID"].transform("count")

df["poly_90d"] = df.groupby("DESYNPUF_ID")["PROD_SRVC_ID"].transform("nunique")

df = df.fillna(0)

# =========================================
# SIDEBAR CONTROLS
# =========================================
st.sidebar.header("🎛 Control Panel")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["XGBoost", "XGBoost Tuned (Best)"]
)

mode = st.sidebar.radio(
    "Patient Selection",
    ["Random Patient", "Manual Selection"]
)

if mode == "Random Patient":
    idx = np.random.randint(0, len(df))
else:
    idx = st.sidebar.slider("Patient Index", 0, len(df)-1, 0)

# MODEL SWITCH
model = xgb if model_choice == "XGBoost" else xgb_tuned

# =========================================
# ✅ FIX: ALIGN FEATURES WITH MODEL
# =========================================
feature_cols = model.get_booster().feature_names

# create missing columns safely
for col in feature_cols:
    if col not in df.columns:
        df[col] = 0

X = df[feature_cols].fillna(0)

patient = df.iloc[idx]
X_patient = X.iloc[[idx]]

# =========================================
# PREDICTION
# =========================================
risk_score = model.predict_proba(X_patient)[0][1]

# =========================================
# TOP DASHBOARD
# =========================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Risk Score", f"{risk_score:.2%}")

with col2:
    st.metric("👤 Age", int(patient.get("age_years", 0)))

with col3:
    st.metric("💊 Total Cost", round(patient.get("TOT_RX_CST_AMT", 0), 2))

with col4:
    st.metric("📦 Days Supply", int(patient.get("DAYS_SUPLY_NUM", 0)))

# =========================================
# RISK ALERT
# =========================================
if risk_score > 0.75:
    st.error("🚨 HIGH RISK: Immediate intervention required")
elif risk_score > 0.45:
    st.warning("⚠️ MEDIUM RISK: Monitor closely")
else:
    st.success("✅ LOW RISK: Patient stable")

# =========================================
# MODEL COMPARISON (FIXED)
# =========================================
st.subheader("⚖️ Model Comparison")

# ==============================
# XGB INPUT
# ==============================
xgb_features = xgb.get_booster().feature_names

for col in xgb_features:
    if col not in df.columns:
        df[col] = 0

X_xgb = df[xgb_features].fillna(0)
X_xgb_patient = X_xgb.iloc[[idx]]

# ==============================
# XGB TUNED INPUT
# ==============================
xgb_tuned_features = xgb_tuned.get_booster().feature_names

for col in xgb_tuned_features:
    if col not in df.columns:
        df[col] = 0

X_xgb_tuned = df[xgb_tuned_features].fillna(0)
X_xgb_tuned_patient = X_xgb_tuned.iloc[[idx]]

# ==============================
# PREDICTIONS
# ==============================
xgb_score = xgb.predict_proba(X_xgb_patient)[0][1]
xgb_tuned_score = xgb_tuned.predict_proba(X_xgb_tuned_patient)[0][1]

# ==============================
# DISPLAY
# ==============================
comparison_df = pd.DataFrame({
    "Model": ["XGB", "XGB Tuned"],
    "Risk Score": [xgb_score, xgb_tuned_score]
})

st.bar_chart(comparison_df.set_index("Model"))

# =========================================
# PATIENT PROFILE
# =========================================
st.subheader("👤 Patient Profile")

colA, colB = st.columns(2)

with colA:
    st.write({
        "Age": patient.get("age_years"),
        "Sex": patient.get("BENE_SEX_IDENT_CD"),
        "Race": patient.get("BENE_RACE_CD")
    })

with colB:
    st.write({
        "Chronic Conditions": sum([patient[c] for c in df.columns if "SP_" in c]),
        "Drug Cost": patient.get("TOT_RX_CST_AMT"),
        "Quantity": patient.get("QTY_DSPNSD_NUM")
    })

# =========================================
# SHAP EXPLANATION
# =========================================
st.subheader("🧠 AI Explanation (Why this risk?)")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

fig, ax = plt.subplots()

shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value,
        data=X.iloc[idx],
        feature_names=feature_cols
    ),
    max_display=10,
    show=False
)

st.pyplot(fig)

# =========================================
# KEY DRIVERS
# =========================================
st.subheader("🔍 Key Risk Drivers")

impact = pd.Series(shap_values[idx], index=feature_cols)
impact = impact.sort_values(key=abs, ascending=False).head(5)

for feat in impact.index:
    direction = "⬆️ increases" if impact[feat] > 0 else "⬇️ decreases"
    st.write(f"• **{feat}** → {direction} risk")

# =========================================
# GLOBAL IMPORTANCE
# =========================================
st.subheader("📈 Global Feature Importance")

importance = model.feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values("importance", ascending=False).head(15)

st.bar_chart(imp_df.set_index("feature"))

# =========================================
# TIMELINE
# =========================================
st.subheader("📅 Patient Timeline")

if "DESYNPUF_ID" in df.columns and "SRVC_DT" in df.columns:
    timeline = df[df["DESYNPUF_ID"] == patient["DESYNPUF_ID"]]

    if len(timeline) > 0:
        fig2, ax2 = plt.subplots(figsize=(10,2))
        ax2.plot(timeline["SRVC_DT"], np.ones(len(timeline)), "o")
        ax2.set_yticks([])
        ax2.set_title("Refill Events Over Time")

        st.pyplot(fig2)

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.markdown("🚀 Built with Explainable AI | Hackathon Ready")