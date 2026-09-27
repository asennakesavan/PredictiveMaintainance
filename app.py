import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Health", page_icon="⚙️", layout="wide", initial_sidebar_state="collapsed")
rf_model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")
with open("metadata_rfs.json", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)
feature_cols, stats = metadata["feature_cols"], metadata["feature_stats"]

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 88% 8%,#193d55 0%,#0b1823 34%,#071019 72%);color:#eaf2f7}#MainMenu,footer,header{visibility:hidden}.block-container{max-width:1180px;padding-top:2.4rem}.eyebrow{color:#80b7c9;font-size:.76rem;font-weight:750;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.45rem}.hero-title{font-size:clamp(2.15rem,4.6vw,3.7rem);font-weight:750;letter-spacing:-.055em;line-height:1;color:#f4f8fa;margin:0}.hero-copy{color:#a8bcc7;font-size:1.03rem;max-width:600px;line-height:1.6;margin-top:.85rem}.live-badge{display:inline-flex;align-items:center;gap:.48rem;border:1px solid #24586a;background:#102a38;color:#9fe4d5;padding:.48rem .8rem;border-radius:99px;font-size:.8rem;font-weight:650;margin-top:1.15rem}.dot{width:.48rem;height:.48rem;border-radius:50%;background:#5bd3b6;box-shadow:0 0 12px #5bd3b6}.section-label{color:#c9dbe3;font-size:.92rem;font-weight:700;margin:.2rem 0 1rem}div[data-testid="stForm"]{background:rgba(15,35,47,.88);border:1px solid #254556;border-radius:18px;padding:1.55rem 1.55rem .85rem}div[data-testid="stNumberInput"],div[data-testid="stSelectbox"]{background:#0b1c28;border:1px solid #284a5d;border-radius:10px;padding:.1rem .5rem}label{color:#c8d8df!important;font-size:.82rem!important;font-weight:650!important}input{color:#f1f7f9!important}.stFormSubmitButton>button{width:100%;border:0;border-radius:10px;padding:.7rem 1rem;background:linear-gradient(90deg,#f5aa49,#ff793f);color:#111a20;font-weight:800;letter-spacing:.02em}.result-card{background:linear-gradient(145deg,#123342,#0c202d);border:1px solid #2a5666;border-radius:18px;padding:1.65rem;min-height:285px}.result-kicker{color:#89b6c4;font-size:.72rem;letter-spacing:.13em;text-transform:uppercase;font-weight:750}.result-value{font-size:2.6rem;font-weight:780;letter-spacing:-.05em;color:#f3f8fa;margin:.55rem 0 .2rem}.result-caption{color:#a4bbc5;line-height:1.5;margin-bottom:1.35rem}.gauge-track{height:11px;background:#06131d;border-radius:99px;overflow:hidden;margin:.45rem 0 .55rem}.gauge-fill{height:100%;border-radius:99px}.risk-low{background:#4ed0ad}.risk-medium{background:#f1b34e}.risk-high{background:#f06b58}.metric-row{display:flex;justify-content:space-between;border-top:1px solid #294958;padding-top:1rem;margin-top:1rem;color:#aec2cb;font-size:.88rem}.metric-row strong{color:#f0f5f7}.info-card{margin-top:1.2rem;background:rgba(14,34,46,.68);border-left:3px solid #5cb8c4;border-radius:8px;padding:.85rem 1rem;color:#a9c0c9;font-size:.86rem;line-height:1.5}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Predictive maintenance platform</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Machine health, made visible.</h1>', unsafe_allow_html=True)
st.markdown('<div class="hero-copy">Enter the latest operating readings to estimate machine-failure risk before it becomes downtime.</div>', unsafe_allow_html=True)
st.markdown('<div class="live-badge"><span class="dot"></span> Random Forest model ready</div>', unsafe_allow_html=True)
st.write("")
form_col, result_col = st.columns([1.45, 1], gap="large")
with form_col:
    st.markdown('<div class="section-label">Operating conditions</div>', unsafe_allow_html=True)
    with st.form("prediction_form"):
        col_a, col_b = st.columns(2)
        with col_a:
            air_temp = st.number_input("Air temperature · K", value=round(stats["Air temperature [K]"]["mean"], 1), format="%.1f")
            rpm = st.number_input("Rotational speed · rpm", value=round(stats["Rotational speed [rpm]"]["mean"], 1), format="%.1f")
            tool_wear = st.number_input("Tool wear · min", value=round(stats["Tool wear [min]"]["mean"], 1), format="%.1f")
        with col_b:
            process_temp = st.number_input("Process temperature · K", value=round(stats["Process temperature [K]"]["mean"], 1), format="%.1f")
            torque = st.number_input("Torque · Nm", value=round(stats["Torque [Nm]"]["mean"], 1), format="%.1f")
            type_choice = st.selectbox("Product quality type", options=metadata["type_categories"], index=1)
        submitted = st.form_submit_button("ANALYZE MACHINE HEALTH")
    st.markdown('<div class="info-card">Tip: use live sensor readings where possible. The result estimates risk, not the specific reason for a potential failure.</div>', unsafe_allow_html=True)

if "prediction" not in st.session_state: st.session_state.prediction = None
if submitted:
    row = {"Air temperature [K]":air_temp,"Process temperature [K]":process_temp,"Rotational speed [rpm]":rpm,"Torque [Nm]":torque,"Tool wear [min]":tool_wear,"Type_L":int(type_choice == "L"),"Type_M":int(type_choice == "M")}
    scaled = scaler.transform(pd.DataFrame([row], columns=feature_cols))
    st.session_state.prediction = int(rf_model.predict(scaled)[0])
    classes = list(rf_model.classes_)
    failure_index = classes.index(1) if 1 in classes else int(np.argmax(classes))
    st.session_state.failure_probability = float(rf_model.predict_proba(scaled)[0][failure_index])

with result_col:
    if st.session_state.prediction is None:
        status, description, risk, style, classification = "Awaiting data", "Complete the operating conditions form to generate a machine-health assessment.", 0, "risk-low", "—"
    else:
        risk = st.session_state.failure_probability
        if risk < .25: status, description, style = "Operating normally", "Continue with standard monitoring. No elevated failure signal was detected.", "risk-low"
        elif risk < .6: status, description, style = "Monitor closely", "An elevated signal was detected. Review this asset during the next planned inspection.", "risk-medium"
        else: status, description, style = "Maintenance advised", "The model indicates a high likelihood of failure. Inspect this asset as soon as practical.", "risk-high"
        classification = "Failure" if st.session_state.prediction else "No failure"
    st.markdown(f'''<div class="result-card"><div class="result-kicker">Risk assessment</div><div class="result-value">{status}</div><div class="result-caption">{description}</div><div class="gauge-track"><div class="gauge-fill {style}" style="width:{risk * 100:.1f}%"></div></div><div class="metric-row"><span>Failure probability</span><strong>{risk:.1%}</strong></div><div class="metric-row"><span>Model classification</span><strong>{classification}</strong></div></div>''', unsafe_allow_html=True)
