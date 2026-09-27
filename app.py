import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Diabetes Prediction App (MLP)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LOAD SAVED ARTIFACTS
# ============================================================
model = keras.models.load_model("diabetes_mlp.keras")
scaler = joblib.load("diabetes_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
impute_medians = joblib.load("impute_medians.pkl")

# ============================================================
# CUSTOM CSS — dark navy/purple gradient theme
# ============================================================
st.markdown("""
<style>
    .stApp {
        background:
            radial-gradient(circle at 8% 5%, rgba(139, 92, 246, 0.10) 0%, transparent 24%),
            radial-gradient(circle at 92% 88%, rgba(59, 130, 246, 0.09) 0%, transparent 25%),
            linear-gradient(135deg, #ffffff 0%, #faf9ff 48%, #f7f9ff 100%);
        color: #172554;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .app-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
        color: #172554;
    }

    .app-title .grad {
        background: linear-gradient(90deg, #9333ea, #6366f1, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .app-subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .badge {
        display: inline-block;
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.10), rgba(59, 130, 246, 0.08));
        border: 1px solid rgba(124, 58, 237, 0.22);
        color: #6d28d9;
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(79, 70, 229, 0.06);
    }

    .hero-heading {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.2;
        color: #172554;
    }

    .hero-heading .grad {
        background: linear-gradient(90deg, #9333ea, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-desc {
        color: #64748b;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.95rem;
    }

    .feature-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }

    .feature-icon {
        width: 38px; height: 38px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
        box-shadow: 0 5px 16px rgba(79, 70, 229, 0.08);
    }

    .feature-title { font-weight: 700; color: #172554; font-size: 0.92rem; }
    .feature-sub { color: #64748b; font-size: 0.78rem; }

    .card {
        background: rgba(255, 255, 255, 0.88);
        border: 1px solid rgba(99, 102, 241, 0.16);
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 12px 35px rgba(30, 41, 59, 0.07);
        margin-bottom: 1.2rem;
    }

    .card-header {
        display: flex; align-items: center; gap: 12px;
        margin-bottom: 1.2rem;
        padding: 0.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.07), rgba(168, 85, 247, 0.05));
    }

    .card-header-icon {
        width: 42px; height: 42px; border-radius: 50%;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 6px 18px rgba(99, 102, 241, 0.20);
    }

    .card-header-title { font-weight: 700; font-size: 1.1rem; color: #172554; }
    .card-header-sub { font-size: 0.8rem; color: #64748b; }

    .field-label {
        display: flex; align-items: center; gap: 8px;
        font-size: 0.85rem; color: #334155; margin-bottom: 4px; font-weight: 600;
    }

    div[data-testid="stNumberInput"] input {
        background: #ffffff !important;
        border: 1px solid #dbe2f0 !important;
        border-radius: 10px !important;
        color: #172554 !important;
        box-shadow: inset 0 1px 2px rgba(15, 23, 42, 0.02);
    }

    div[data-testid="stNumberInput"] input:focus {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.10) !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #f5f3ff !important;
        border: 1px solid #e0e7ff !important;
        color: #4f46e5 !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #ede9fe !important;
        color: #4338ca !important;
    }

    div[data-testid="stButton"] button {
        width: 100%;
        background: linear-gradient(90deg, #a855f7, #6366f1, #3b82f6);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 0;
        margin-top: 0.5rem;
        box-shadow: 0 8px 20px rgba(99, 102, 241, 0.20);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px);
        border: none;
        color: white;
        box-shadow: 0 10px 24px rgba(99, 102, 241, 0.28);
    }

    .about-item { display: flex; gap: 12px; margin-bottom: 16px; align-items: flex-start; }

    .about-icon {
        width: 34px; height: 34px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.06);
    }

    .about-title { font-weight: 700; font-size: 0.88rem; color: #172554; }
    .about-sub { font-size: 0.76rem; color: #64748b; }

    .quote-box {
        background: linear-gradient(90deg, rgba(168, 85, 247, 0.08), rgba(99, 102, 241, 0.06));
        border-left: 3px solid #a855f7;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-style: italic;
        color: #5b21b6;
        font-size: 0.9rem;
        box-shadow: 0 8px 22px rgba(99, 102, 241, 0.06);
    }

    .result-diabetic {
        background: rgba(239, 68, 68, 0.07);
        border: 1px solid rgba(239, 68, 68, 0.28);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-nondiabetic {
        background: rgba(34, 197, 94, 0.07);
        border: 1px solid rgba(34, 197, 94, 0.28);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }

    .result-label { font-size: 1.4rem; font-weight: 800; margin-bottom: 4px; }
    .result-prob { font-size: 0.95rem; color: #475569; }

    .footer-tag {
        text-align: center; color: #94a3b8; font-size: 0.85rem; margin-top: 2rem;
        letter-spacing: 2px;
    }

    /* Streamlit alerts/messages */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        border: 1px solid #f2d675 !important;
        background: #fffbea !important;
        color: #713f12 !important;
    }

    div[data-testid="stAlert"] *,
    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span {
        color: #713f12 !important;
    }

    div[data-testid="stAlert"] svg {
        color: #d97706 !important;
        fill: #d97706 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-top: 0.5rem;">
    <div class="app-title">🩺 Diabetes <span class="grad">Prediction App (MLP)</span></div>
    <div class="app-subtitle">Enter patient details below to predict the likelihood of diabetes using a<br>Machine Learning model (Multilayer Perceptron).</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT: 3 columns — left hero / center form / right about
# ============================================================
left, center, right = st.columns([1, 1.6, 1])

# ---------------- LEFT COLUMN ----------------
with left:
    st.markdown("""
    <div class="badge">Early Prediction &nbsp;•&nbsp; Healthier Tomorrow</div>
    <div class="hero-heading">Predict Diabetes,<br><span class="grad">Take Control.</span></div>
    <div class="hero-desc">Use machine learning to assess diabetes risk and make informed health decisions.</div>

    <div class="feature-row">
        <div class="feature-icon" style="background:rgba(168,85,247,0.10); color:#7e22ce;">⚡</div>
        <div><div class="feature-title">Fast &amp; Accurate</div><div class="feature-sub">ML-based prediction</div></div>
    </div>
    <div class="feature-row">
        <div class="feature-icon" style="background:rgba(34,197,94,0.10); color:#16a34a;">🛡️</div>
        <div><div class="feature-title">Your Data Stays Private</div><div class="feature-sub">Safe &amp; secure</div></div>
    </div>
    <div class="feature-row">
        <div class="feature-icon" style="background:rgba(236,72,153,0.10); color:#db2777;">❤️</div>
        <div><div class="feature-title">A Healthier You</div><div class="feature-sub">Awareness leads to better choices</div></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- CENTER COLUMN (FORM) ----------------
with center:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon">👤</div>
            <div>
                <div class="card-header-title">Patient Information</div>
                <div class="card-header-sub">Adjust the values using the controls below</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="field-label">🤰 Pregnancies</div>', unsafe_allow_html=True)
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, label_visibility="collapsed")

        st.markdown('<div class="field-label">💓 Blood Pressure</div>', unsafe_allow_html=True)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70, label_visibility="collapsed")

        st.markdown('<div class="field-label">💉 Insulin</div>', unsafe_allow_html=True)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, label_visibility="collapsed")

        st.markdown('<div class="field-label">📄 Diabetes Pedigree Function</div>', unsafe_allow_html=True)
        dpf = st.number_input("DPF", min_value=0.0, max_value=3.0, value=0.5, label_visibility="collapsed")

    with c2:
        st.markdown('<div class="field-label">🩸 Glucose</div>', unsafe_allow_html=True)
        glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120, label_visibility="collapsed")

        st.markdown('<div class="field-label">🧬 Skin Thickness</div>', unsafe_allow_html=True)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, label_visibility="collapsed")

        st.markdown('<div class="field-label">⚖️ BMI</div>', unsafe_allow_html=True)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, label_visibility="collapsed")

        st.markdown('<div class="field-label">📅 Age</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=30, label_visibility="collapsed")

    predict_clicked = st.button("⚡ Predict")

    st.markdown("</div>", unsafe_allow_html=True)  # close .card

    if predict_clicked:
        if glucose == 0 or blood_pressure == 0 or bmi == 0:
            st.warning("⚠️ Glucose, Blood Pressure, and BMI cannot be 0. Please enter valid values.")
        else:
            # Replicate training-time imputation for zero-coded missing values
            skin_v = skin_thickness if skin_thickness != 0 else impute_medians['SkinThickness']
            insulin_v = insulin if insulin != 0 else impute_medians['Insulin']

            input_dict = {
                'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
                'SkinThickness': skin_v, 'Insulin': insulin_v, 'BMI': bmi,
            'DiabetesPedigreeFunction': dpf, 'Age': age
            }
            df_input = pd.DataFrame([input_dict])

            age_group = pd.cut(df_input['Age'], bins=[0, 30, 40, 50, 60, 200],
                labels=['<=30', '31-40', '41-50', '51-60', '60+'])
            bmi_cat = 'Underweight' if bmi < 18.5 else 'Normal' if bmi < 25 else 'Overweight' if bmi < 30 else 'Obese'
            gluc_cat = 'Normal' if glucose < 100 else 'Prediabetic' if glucose < 126 else 'Diabetic_Range'

            df_input['Glucose_BMI_Interaction'] = glucose * bmi
            df_input['Age_Pregnancies_Interaction'] = age * pregnancies
            df_input['Insulin_log'] = np.log1p(insulin_v)
            df_input['DPF_log'] = np.log1p(dpf)

            for col in feature_columns:
                if col.startswith('AgeGroup_'):
                    df_input[col] = (age_group.astype(str) == col.replace('AgeGroup_', '')).astype(int)
                elif col.startswith('BMICategory_'):
                    df_input[col] = int(bmi_cat == col.replace('BMICategory_', ''))
                elif col.startswith('GlucoseCategory_'):
                    df_input[col] = int(gluc_cat == col.replace('GlucoseCategory_', ''))

            df_input = df_input.reindex(columns=feature_columns, fill_value=0)

            # scaler was fit only on continuous columns; one-hot columns stay unscaled
            continuous_cols = list(scaler.feature_names_in_)
            onehot_cols = [c for c in feature_columns if c not in continuous_cols]

            scaled_cont = pd.DataFrame(
                scaler.transform(df_input[continuous_cols]),
                columns=continuous_cols, index=df_input.index
)
            input_scaled_df = pd.concat([scaled_cont, df_input[onehot_cols]], axis=1)[feature_columns]
            input_scaled = input_scaled_df.values

            prob = model.predict(input_scaled, verbose=0)[0][0]
            prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

            if prediction == "Diabetic":
                st.markdown(f"""
                <div class="result-diabetic">
                    <div class="result-label" style="color:#dc2626;">⚠️ Prediction: Diabetic</div>
                    <div class="result-prob">Probability: {prob*100:.2f}%</div>
                </div>
            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-nondiabetic">
                    <div class="result-label" style="color:#16a34a;">✅ Prediction: Non-Diabetic</div>
                    <div class="result-prob">Probability: {prob*100:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

# ---------------- RIGHT COLUMN ----------------
with right:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <div class="card-header-icon" style="background:linear-gradient(135deg,#3b82f6,#6366f1); box-shadow:0 6px 18px rgba(59,130,246,0.20);">ℹ️</div>
            <div class="card-header-title">About This App</div>
        </div>
        <div style="color:#64748b; font-size:0.85rem; margin-bottom:1.2rem;">
            This app uses a Multilayer Perceptron (MLP) model trained on the Pima Indians Diabetes dataset
            to predict the likelihood of diabetes based on key health metrics.
        </div>
        <hr style="border-color: #e2e8f0; margin: 1rem 0;">
        <div class="about-item">
            <div class="about-icon" style="background:rgba(59,130,246,0.10); color:#2563eb;">📊</div>
            <div><div class="about-title">MLP (Neural Network)</div><div class="about-sub">Powered by deep learning</div></div>
        </div>
        <div class="about-item">
            <div class="about-icon" style="background:rgba(99,102,241,0.10); color:#4f46e5;">🗄️</div>
            <div><div class="about-title">Pima Indians Dataset</div><div class="about-sub">Trusted &amp; widely used</div></div>
        </div>
        <div class="about-item">
            <div class="about-icon" style="background:rgba(239,68,68,0.10); color:#dc2626;">🎯</div>
            <div><div class="about-title">For Educational Use</div><div class="about-sub">Not a medical diagnosis</div></div>
        </div>
    </div>
    <div class="quote-box">
        “Awareness today for a healthier tomorrow.”
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-tag">— &nbsp; C A R E &nbsp;•&nbsp; P R E D I C T &nbsp;•&nbsp; P R E V E N T &nbsp; —</div>
""", unsafe_allow_html=True)