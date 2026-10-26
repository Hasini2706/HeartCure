import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import base64
import os
import textwrap

# -----------------------------------------------------------------------------
# Page Configuration & Styling (HeartCare Clean Medical Design)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeartCare | Heart Disease Prediction Using SVM",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load local assets as Base64
def get_base64_image(filename):
    p = os.path.join(os.path.dirname(__file__), "assets", filename)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

def render_html(html_str):
    cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)

heart_3d_b64 = get_base64_image("heart_3d.jpg")
doctor_b64 = get_base64_image("doctor.jpg")

# Inject Custom CSS matching the HeartCare mockup exactly
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Light, warm medical background */
    .stApp {
        background-color: #faf5f5;
        color: #1e293b;
    }

    /* Top Navigation Bar */
    .top-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 14px 28px;
        background: #ffffff;
        border-bottom: 1px solid #f1e5e5;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 2px 10px rgba(197, 34, 34, 0.03);
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.4rem;
        font-weight: 800;
        color: #c52222;
        letter-spacing: -0.02em;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #f1e5e5;
        padding-top: 10px;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.35rem;
        font-weight: 800;
        color: #c52222;
        padding: 10px 14px 24px 14px;
        letter-spacing: -0.02em;
    }

    /* Custom Cards */
    .content-card, div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #f1e5e5 !important;
        border-radius: 20px !important;
        padding: 28px !important;
        box-shadow: 0 4px 20px rgba(197, 34, 34, 0.04) !important;
        margin-bottom: 24px !important;
    }

    /* Highlight badge */
    .sub-badge {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #c52222;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-heading {
        font-size: 3rem;
        font-weight: 800;
        color: #111827;
        line-height: 1.15;
        margin-bottom: 16px;
        letter-spacing: -0.03em;
    }

    .hero-heading span.svm-highlight {
        color: #c52222;
    }

    .hero-desc {
        font-size: 1.05rem;
        color: #4b5563;
        line-height: 1.6;
        max-width: 580px;
        margin-bottom: 28px;
    }

    /* Feature Highlights in Home Page */
    .feature-card {
        background: #ffffff;
        border: 1px solid #f3e8e8;
        border-radius: 16px;
        padding: 18px 22px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03);
    }

    .feature-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: #fdf2f2;
        color: #c52222;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
    }

    /* Prediction Result Styles */
    .result-badge-card {
        border-radius: 20px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }

    .result-badge-card.low-risk {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }

    .result-badge-card.mod-risk {
        background: #fffbeb;
        border: 1px solid #fde68a;
    }

    .result-badge-card.high-risk {
        background: #fef2f2;
        border: 1px solid #fecaca;
    }

    .status-icon-circle {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 16px auto;
        font-size: 2rem;
    }

    .low-risk .status-icon-circle {
        background: #22c55e;
        color: #ffffff;
    }

    .mod-risk .status-icon-circle {
        background: #f59e0b;
        color: #ffffff;
    }

    .high-risk .status-icon-circle {
        background: #ef4444;
        color: #ffffff;
    }

    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .low-risk .result-title { color: #166534; }
    .mod-risk .result-title { color: #92400e; }
    .high-risk .result-title { color: #991b1b; }

    .result-subtitle {
        font-size: 0.95rem;
        color: #4b5563;
        margin-bottom: 0;
    }

    /* Details Table Card */
    .details-table-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }

    .details-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #f3f4f6;
        font-size: 0.95rem;
    }
    .details-row:last-child {
        border-bottom: none;
    }

    .pill-status {
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .pill-green { background: #dcfce7; color: #15803d; }
    .pill-amber { background: #fef3c7; color: #b45309; }
    .pill-red { background: #fee2e2; color: #b91c1c; }

    /* Advisory Box */
    .advisory-box {
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 0.88rem;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }
    .advisory-info { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .advisory-warn { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .advisory-danger { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }

    /* Custom Streamlit Buttons */
    div.stButton > button {
        border-radius: 30px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.2s ease !important;
    }

    /* Primary button style */
    div.stButton > button[kind="primary"] {
        background: #c52222 !important;
        border: 1px solid #c52222 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(197, 34, 34, 0.25) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #a91b1b !important;
        box-shadow: 0 6px 18px rgba(197, 34, 34, 0.35) !important;
        transform: translateY(-1px);
    }

    /* Secondary button style */
    div.stButton > button[kind="secondary"] {
        background: #ffffff !important;
        border: 1.5px solid #c52222 !important;
        color: #c52222 !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background: #fdf2f2 !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model & Scaler Artifact Management
# -----------------------------------------------------------------------------
@st.cache_resource
def load_clinical_artifacts():
    try:
        model = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception:
        return None, None

model, scaler = load_clinical_artifacts()

if model is None or scaler is None:
    st.error("🚨 Clinical model artifacts (`svm_model.pkl`, `scaler.pkl`) could not be loaded. Please ensure the model training script has run.")
    st.stop()

# -----------------------------------------------------------------------------
# Navigation & Session State
# -----------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

if "history" not in st.session_state:
    st.session_state["history"] = [
        {"datetime": "12 Oct 2026, 10:15 AM", "age": 45, "sex": "Male", "prediction": "No Disease", "confidence": "92.4%", "badge": "pill-green"},
        {"datetime": "10 Oct 2026, 02:30 PM", "age": 58, "sex": "Male", "prediction": "Heart Disease", "confidence": "88.7%", "badge": "pill-red"},
        {"datetime": "08 Oct 2026, 11:20 AM", "age": 50, "sex": "Female", "prediction": "Possible Risk", "confidence": "76.3%", "badge": "pill-amber"},
        {"datetime": "06 Oct 2026, 04:10 PM", "age": 39, "sex": "Female", "prediction": "No Disease", "confidence": "95.1%", "badge": "pill-green"},
        {"datetime": "02 Oct 2026, 09:45 AM", "age": 60, "sex": "Male", "prediction": "Heart Disease", "confidence": "83.5%", "badge": "pill-red"}
    ]

# Default form state
if "form_data" not in st.session_state:
    st.session_state["form_data"] = {
        "age": 45, "sex": "Male", "cp": "Typical Angina", "trestbps": 120, "chol": 200,
        "fbs": "No", "restecg": "Normal", "thalach": 150, "exang": "No",
        "oldpeak": 1.0, "slope": "Upsloping", "ca": 0, "thal": "Normal"
    }

# -----------------------------------------------------------------------------
# Sidebar Navigation (Matching Mockup)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#c52222" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            <path d="M12 5v4l3 2"/>
        </svg>
        HeartCare
    </div>
    """, unsafe_allow_html=True)

    nav_pages = ["Home", "Predict", "History", "About"]
    icons = {"Home": "🏠", "Predict": "🩺", "History": "🕒", "About": "ℹ️"}

    for p in nav_pages:
        is_active = (st.session_state["page"] == p) or (st.session_state["page"] == "Result" and p == "Predict")
        btn_type = "primary" if is_active else "secondary"
        if st.button(f"{icons[p]}  {p}", key=f"nav_{p}", use_container_width=True, type=btn_type):
            st.session_state["page"] = p
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.caption("HeartCare AI Diagnostic Engine")
    st.caption("Powered by Support Vector Machine (SVM)")

# =============================================================================
# PAGE 1: HOME
# =============================================================================
if st.session_state["page"] == "Home":
    st.markdown("""
    <div class="top-navbar">
        <div class="brand-logo">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#c52222" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>
            </svg>
            HeartCare
        </div>
        <div style="display:flex; align-items:center; gap:20px; font-weight:600; font-size:0.95rem; color:#4b5563;">
            <span style="color:#c52222; cursor:pointer;">Home</span>
            <span style="cursor:pointer;">Predict</span>
            <span style="cursor:pointer;">History</span>
            <span style="cursor:pointer;">About</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    hero_col1, hero_col2 = st.columns([1.1, 0.9], gap="large")

    with hero_col1:
        st.markdown("""
        <div class="sub-badge">AI POWERED HEALTH INSIGHTS</div>
        <h1 class="hero-heading">Heart Disease<br>Prediction Using <span class="svm-highlight">SVM</span></h1>
        <p class="hero-desc">
            A machine learning based web application to predict the risk of heart disease using Support Vector Machine (SVM). Enter your health details and get an instant prediction.
        </p>
        """, unsafe_allow_html=True)

        cta_col1, cta_col2, cta_col3 = st.columns([1.2, 1.2, 1.5])
        with cta_col1:
            if st.button("Predict Now  ➔", key="hero_predict_btn", type="primary", use_container_width=True):
                st.session_state["page"] = "Predict"
                st.rerun()
        with cta_col2:
            if st.button("Learn More", key="hero_learn_btn", type="secondary", use_container_width=True):
                st.session_state["page"] = "About"
                st.rerun()

    with hero_col2:
        if heart_3d_b64:
            st.markdown(f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{heart_3d_b64}" style="max-width: 85%; height: auto; border-radius: 20px; box-shadow: 0 10px 30px rgba(197, 34, 34, 0.12);" />
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Bottom 3 Feature Highlights
    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
    with f_col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">📈</div>
            <div>
                <div style="font-weight:700; font-size:1.05rem; color:#111827;">Fast Prediction</div>
                <div style="font-size:0.85rem; color:#6b7280; margin-top:2px;">Get results in seconds</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">🛡️</div>
            <div>
                <div style="font-weight:700; font-size:1.05rem; color:#111827;">ML Powered</div>
                <div style="font-size:0.85rem; color:#6b7280; margin-top:2px;">Using SVM Algorithm</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-box">❤️</div>
            <div>
                <div style="font-weight:700; font-size:1.05rem; color:#111827;">User Friendly</div>
                <div style="font-size:0.85rem; color:#6b7280; margin-top:2px;">Simple and clean interface</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE 2: PREDICT (PATIENT INFORMATION FORM)
# =============================================================================
elif st.session_state["page"] == "Predict":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #111827; margin: 0;">Patient Information</h2>
        <p style="font-size: 0.95rem; color: #6b7280; margin-top: 4px;">Enter the following details to predict the risk of heart disease using SVM.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        form_col1, form_col2 = st.columns(2, gap="large")

        # Column 1
        with form_col1:
            age = st.number_input("👤 Age (years)", min_value=1, max_value=120, value=st.session_state["form_data"]["age"])
            
            cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
            cp_choice = st.selectbox("🫀 Chest Pain Type", list(cp_map.keys()), index=0)
            
            chol = st.number_input("🩸 Cholesterol (mg/dl)", min_value=50, max_value=700, value=st.session_state["form_data"]["chol"])
            
            restecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg_choice = st.selectbox("📈 Resting ECG", list(restecg_map.keys()), index=0)
            
            exang_map = {"No": 0, "Yes": 1}
            exang_choice = st.selectbox("🏃 Exercise Induced Angina", list(exang_map.keys()), index=0)
            
            slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope_choice = st.selectbox("📐 Slope of ST Segment", list(slope_map.keys()), index=0)
            
            thal_map = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2, "Other": 3}
            thal_choice = st.selectbox("🧬 Thalassemia", list(thal_map.keys()), index=0)

        # Column 2
        with form_col2:
            sex_choice = st.selectbox("👥 Sex", ["Male", "Female"], index=0)
            sex_val = 1 if sex_choice == "Male" else 0
            
            trestbps = st.number_input("🩺 Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=st.session_state["form_data"]["trestbps"])
            
            fbs_choice = st.selectbox("🧪 Fasting Blood Sugar (> 120 mg/dl)", ["No", "Yes"], index=0)
            fbs_val = 1 if fbs_choice == "Yes" else 0
            
            thalach = st.number_input("❤️ Maximum Heart Rate", min_value=50, max_value=250, value=st.session_state["form_data"]["thalach"])
            
            oldpeak = st.number_input("📉 ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=float(st.session_state["form_data"]["oldpeak"]), step=0.1, format="%.1f")
            
            ca = st.selectbox("🔬 Number of Major Vessels (0-3)", [0, 1, 2, 3, 4], index=st.session_state["form_data"]["ca"])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_submit = st.button("🩺 Predict Heart Disease Risk", type="primary", use_container_width=True)

    if predict_submit:
        # Collect 13 features in correct order
        features = np.array([[
            age,
            sex_val,
            cp_map[cp_choice],
            trestbps,
            chol,
            fbs_val,
            restecg_map[restecg_choice],
            thalach,
            exang_map[exang_choice],
            oldpeak,
            slope_map[slope_choice],
            ca,
            thal_map[thal_choice]
        ]])

        scaled_features = scaler.transform(features)
        pred = model.predict(scaled_features)[0]
        decision = model.decision_function(scaled_features)[0]
        prob = 1.0 / (1.0 + np.exp(-decision))
        conf_pct = round(prob * 100, 1) if pred == 1 else round((1.0 - prob) * 100, 1)

        # Determine Risk Tier
        if prob >= 0.60:
            tier = "High Risk"
            badge = "pill-red"
            pred_text = "Heart Disease"
        elif prob >= 0.40:
            tier = "Moderate Risk"
            badge = "pill-amber"
            pred_text = "Possible Risk"
        else:
            tier = "Low Risk"
            badge = "pill-green"
            pred_text = "No Disease"

        # Save to history
        new_entry = {
            "datetime": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "age": age,
            "sex": sex_choice,
            "prediction": pred_text,
            "confidence": f"{conf_pct}%",
            "badge": badge
        }
        st.session_state["history"].insert(0, new_entry)

        # Save result state and redirect
        st.session_state["last_result"] = {
            "tier": tier,
            "confidence": f"{conf_pct}%",
            "prediction": pred_text
        }
        st.session_state["page"] = "Result"
        st.rerun()

# =============================================================================
# PAGE 3: PREDICTION RESULT (3 MOCKUP STATES: LOW, MODERATE, HIGH RISK)
# =============================================================================
elif st.session_state["page"] == "Result":
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #111827; margin: 0;">Prediction Result</h2>
    </div>
    """, unsafe_allow_html=True)

    result_data = st.session_state.get("last_result", {
        "tier": "Low Risk",
        "confidence": "92.4%",
        "prediction": "No Disease"
    })

    tier = result_data["tier"]

    res_card_class = "low-risk"
    icon_char = "✓"
    sub_msg = "The model predicts that you are less likely to have heart disease."
    advisory_class = "advisory-info"
    advisory_msg = "Keep maintaining a healthy lifestyle. Regular exercise and balanced diet are recommended."
    pill_class = "pill-green"
    badge_label = "No Heart Disease"

    if tier == "Moderate Risk":
        res_card_class = "mod-risk"
        icon_char = "!"
        sub_msg = "The model predicts that you may have a moderate risk of heart disease."
        advisory_class = "advisory-warn"
        advisory_msg = "Consider consulting a doctor for further evaluation and lifestyle guidance."
        pill_class = "pill-amber"
        badge_label = "Possible Heart Disease"
    elif tier == "High Risk":
        res_card_class = "high-risk"
        icon_char = "!"
        sub_msg = "The model predicts that you are likely to have heart disease."
        advisory_class = "advisory-danger"
        advisory_msg = "Please consult a cardiologist for detailed medical checkup. This is an AI-based prediction and not a final diagnosis."
        pill_class = "pill-red"
        badge_label = "Heart Disease"

    with st.container():
        result_html = f"""<div class="content-card">
<div class="result-badge-card {res_card_class}">
<div class="status-icon-circle">{icon_char}</div>
<div class="result-title">{tier}</div>
<p class="result-subtitle">{sub_msg}</p>
</div>
<div class="details-table-card">
<div style="font-weight:700; color:#111827; margin-bottom:12px; font-size:1.05rem;">Prediction Details</div>
<div class="details-row">
<span style="color:#6b7280;">Prediction</span>
<span class="pill-status {pill_class}">{badge_label}</span>
</div>
<div class="details-row">
<span style="color:#6b7280;">Confidence</span>
<span style="font-weight:700; color:#111827;">{result_data['confidence']}</span>
</div>
<div class="details-row">
<span style="color:#6b7280;">Model Used</span>
<span style="font-weight:600; color:#374151;">Support Vector Machine (SVM)</span>
</div>
</div>
<div class="advisory-box {advisory_class}">
<span style="font-size:1.2rem;">ℹ️</span>
<span>{advisory_msg}</span>
</div>
</div>"""
        render_html(result_html)

        btn_col1, btn_col2, btn_col3 = st.columns([1.2, 1, 2])
        with btn_col1:
            if st.button("Check Another Prediction", key="another_pred_btn", type="primary", use_container_width=True):
                st.session_state["page"] = "Predict"
                st.rerun()
        with btn_col2:
            if st.button("View History", key="view_hist_btn", type="secondary", use_container_width=True):
                st.session_state["page"] = "History"
                st.rerun()

# =============================================================================
# PAGE 4: PREDICTION HISTORY
# =============================================================================
elif st.session_state["page"] == "History":
    hist_top1, hist_top2 = st.columns([3, 1])
    with hist_top1:
        st.markdown("""
        <div>
            <h2 style="font-size: 1.85rem; font-weight: 800; color: #111827; margin: 0;">Prediction History</h2>
        </div>
        """, unsafe_allow_html=True)
    with hist_top2:
        if st.button("Clear History", key="clear_hist_btn", type="secondary", use_container_width=True):
            st.session_state["history"] = []
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    with st.container():
        if len(st.session_state["history"]) == 0:
            render_html("""
            <div class="content-card" style="text-align:center; padding:48px 24px; color:#6b7280;">
                <div style="font-size:2.5rem; margin-bottom:12px;">📋</div>
                <div style="font-size:1.1rem; font-weight:700; color:#111827;">No predictions yet</div>
                <div style="font-size:0.9rem; margin-top:4px;">Run a prediction to start tracking patient history.</div>
            </div>
            """)
        else:
            # Render Clean History Table
            rows_html = ""
            for item in st.session_state["history"]:
                rows_html += f"""
                <tr style="border-bottom: 1px solid #f1f5f9;">
                    <td style="padding: 14px 16px; color:#374151; font-size:0.9rem;">{item['datetime']}</td>
                    <td style="padding: 14px 16px; color:#374151; font-size:0.9rem;">{item['age']}</td>
                    <td style="padding: 14px 16px; color:#374151; font-size:0.9rem;">{item['sex']}</td>
                    <td style="padding: 14px 16px;"><span class="pill-status {item['badge']}">{item['prediction']}</span></td>
                    <td style="padding: 14px 16px; font-weight:700; color:#111827; font-size:0.9rem;">{item['confidence']}</td>
                </tr>
                """

            table_html = f"""
            <div class="content-card" style="overflow-x:auto;">
                <table style="width:100%; border-collapse:collapse; text-align:left;">
                    <thead>
                        <tr style="border-bottom: 2px solid #e5e7eb; color:#4b5563; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.04em;">
                            <th style="padding: 12px 16px;">Date & Time</th>
                            <th style="padding: 12px 16px;">Age</th>
                            <th style="padding: 12px 16px;">Sex</th>
                            <th style="padding: 12px 16px;">Prediction</th>
                            <th style="padding: 12px 16px;">Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """
            render_html(table_html)

# =============================================================================
# PAGE 5: ABOUT THE PROJECT
# =============================================================================
elif st.session_state["page"] == "About":
    render_html("""
    <div style="margin-bottom: 24px;">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #111827; margin: 0;">About the Project</h2>
    </div>
    """)

    with st.container():
        about_card_html = """
        <div class="content-card">
            <p style="font-size: 1rem; color: #4b5563; line-height: 1.65; margin-bottom: 24px;">
                This project uses <strong>Support Vector Machine (SVM)</strong>, a supervised machine learning algorithm, to predict the risk of heart disease based on various medical parameters. The model is trained on a standard heart disease dataset and provides an early indication of risk, which can help in timely consultation and preventive care.
            </p>

            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin-bottom:12px;">
                <div style="background:#fdf2f2; border:1px solid #fecaca; border-radius:14px; padding:16px; display:flex; align-items:center; gap:12px;">
                    <span style="font-size:1.4rem;">❤️</span>
                    <div>
                        <div style="font-weight:700; color:#111827; font-size:0.95rem;">Machine Learning</div>
                        <div style="font-size:0.8rem; color:#6b7280;">Support Vector Machine</div>
                    </div>
                </div>

                <div style="background:#fdf2f2; border:1px solid #fecaca; border-radius:14px; padding:16px; display:flex; align-items:center; gap:12px;">
                    <span style="font-size:1.4rem;">🩺</span>
                    <div>
                        <div style="font-weight:700; color:#111827; font-size:0.95rem;">Health Parameters</div>
                        <div style="font-size:0.8rem; color:#6b7280;">13 medical attributes</div>
                    </div>
                </div>

                <div style="background:#fdf2f2; border:1px solid #fecaca; border-radius:14px; padding:16px; display:flex; align-items:center; gap:12px;">
                    <span style="font-size:1.4rem;">🎯</span>
                    <div>
                        <div style="font-weight:700; color:#111827; font-size:0.95rem;">Fast & Accurate</div>
                        <div style="font-size:0.8rem; color:#6b7280;">Reliable predictions</div>
                    </div>
                </div>

                <div style="background:#fdf2f2; border:1px solid #fecaca; border-radius:14px; padding:16px; display:flex; align-items:center; gap:12px;">
                    <span style="font-size:1.4rem;">💡</span>
                    <div>
                        <div style="font-weight:700; color:#111827; font-size:0.95rem;">Easy to Use</div>
                        <div style="font-size:0.8rem; color:#6b7280;">Simple web interface</div>
                    </div>
                </div>
            </div>
        </div>
        """
        render_html(about_card_html)

    # Doctor Illustration & Important Note Card
    about_col1, about_col2 = st.columns([1, 2], gap="large")

    with about_col1:
        if doctor_b64:
            doctor_html = f"""
            <div style="text-align: center;">
                <img src="data:image/jpeg;base64,{doctor_b64}" style="max-width: 80%; height: auto; border-radius: 20px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);" />
            </div>
            """
            render_html(doctor_html)

    with about_col2:
        note_html = """
        <div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 16px; padding: 24px; height: 100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="display:flex; align-items:center; gap:8px; font-weight:800; font-size:1.1rem; color:#1e40af; margin-bottom:8px;">
                <span>ℹ️</span> Important Note
            </div>
            <p style="color:#1e3a8a; font-size:0.95rem; line-height:1.6; margin:0;">
                This is an AI-based prediction tool developed for educational purposes only. It is not a substitute for professional medical advice, diagnosis or treatment. Please consult a qualified healthcare professional for accurate diagnosis and guidance.
            </p>
        </div>
        """
        render_html(note_html)