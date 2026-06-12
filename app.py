import streamlit as st
import joblib
import pandas as pd

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp { background: #F7F8FC; color: #1A1D23; font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0F1623 !important;
    border-right: 1px solid #1E2533;
}
[data-testid="stSidebar"] * { color: #C9D1E0 !important; }
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown strong { color: #FFFFFF !important; }
[data-testid="stSidebar"] hr { border-color: #1E2533 !important; margin: 1rem 0; }

/* ── Main layout ── */
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1100px; }

/* ── Page header ── */
.page-header {
    display: flex; align-items: flex-end; justify-content: space-between;
    border-bottom: 1px solid #E2E6EF; padding-bottom: 1.25rem; margin-bottom: 2rem;
}
.page-header-left h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; font-weight: 400; color: #0F1623; margin: 0; line-height: 1.2;
}
.page-header-left p {
    font-size: 0.875rem; color: #6B7280; margin: 0.35rem 0 0;
}
.badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; padding: 4px 10px;
    border-radius: 4px; background: #EEF2FF; color: #4F46E5;
}

/* ── Disease tabs ── */
.disease-tabs {
    display: flex; gap: 0; margin-bottom: 2rem;
    border: 1px solid #E2E6EF; border-radius: 8px; overflow: hidden;
}
.disease-tab {
    flex: 1; padding: 0.75rem 1rem; text-align: center;
    font-size: 0.875rem; font-weight: 500; cursor: pointer;
    background: white; color: #6B7280; border: none;
    transition: all 0.15s;
}
.disease-tab.active { background: #0F1623; color: white; }

/* ── Section card ── */
.section-card {
    background: white; border: 1px solid #E2E6EF;
    border-radius: 10px; padding: 1.75rem 2rem; margin-bottom: 1.25rem;
}
.section-card-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #9CA3AF; margin-bottom: 1.25rem;
    padding-bottom: 0.75rem; border-bottom: 1px solid #F3F4F6;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stRadio"] label {
    font-size: 0.825rem !important; font-weight: 500 !important;
    color: #374151 !important; margin-bottom: 2px;
}
div[data-testid="stSlider"] { margin-bottom: 0.25rem; }

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: #0F1623 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-size: 0.875rem !important; font-weight: 600 !important;
    padding: 0.65rem 1.5rem !important; letter-spacing: 0.02em;
    transition: background 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    background: #1E2D45 !important;
}

/* ── Result panels ── */
.result-panel {
    border-radius: 10px; padding: 1.5rem 1.75rem; margin-top: 1.5rem;
    border: 1px solid transparent;
}
.result-high {
    background: #FFF7F7; border-color: #FECACA;
}
.result-low {
    background: #F0FDF4; border-color: #BBF7D0;
}
.result-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 0.5rem;
}
.result-icon { font-size: 1.2rem; }
.result-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-high .result-label { color: #DC2626; }
.result-low  .result-label { color: #16A34A; }
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem; font-weight: 400; margin: 0 0 0.4rem;
}
.result-high .result-title { color: #991B1B; }
.result-low  .result-title { color: #166534; }
.result-desc { font-size: 0.875rem; color: #4B5563; margin: 0; }

/* ── Precautions ── */
.precaution-panel {
    background: #F8FAFF; border: 1px solid #DBEAFE;
    border-radius: 8px; padding: 1.25rem 1.5rem; margin-top: 1rem;
}
.precaution-title {
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #3B82F6; margin-bottom: 0.75rem;
}
.precaution-list { margin: 0; padding: 0; list-style: none; }
.precaution-list li {
    font-size: 0.85rem; color: #1E3A5F; padding: 0.3rem 0;
    padding-left: 1.1rem; position: relative; line-height: 1.5;
}
.precaution-list li::before {
    content: "→"; position: absolute; left: 0;
    color: #3B82F6; font-size: 0.8rem;
}

/* ── Probability metric ── */
.prob-block {
    display: inline-flex; flex-direction: column;
    background: white; border: 1px solid #E2E6EF;
    border-radius: 8px; padding: 0.85rem 1.25rem; margin-top: 1.25rem;
}
.prob-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9CA3AF; }
.prob-value { font-size: 2rem; font-weight: 700; color: #0F1623; line-height: 1.1; }
.prob-sub { font-size: 0.75rem; color: #6B7280; margin-top: 2px; }

/* ── Disclaimer ── */
.disclaimer {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 8px; padding: 0.75rem 1rem;
    font-size: 0.8rem; color: #92400E; margin-top: 2rem;
    display: flex; gap: 8px; align-items: flex-start;
}

/* ── Sidebar brand ── */
.brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem; color: #FFFFFF !important;
    line-height: 1.2; margin-bottom: 2px;
}
.brand-sub { font-size: 0.78rem; color: #6B7A99 !important; }
.nav-pill {
    display: flex; align-items: center; gap: 8px;
    padding: 0.55rem 0.75rem; border-radius: 6px;
    font-size: 0.85rem; font-weight: 500;
    color: #C9D1E0 !important; margin-bottom: 2px;
    transition: background 0.12s;
}
.nav-pill:hover { background: rgba(255,255,255,0.06); }
.meta-row { font-size: 0.78rem; color: #4B5A72 !important; }

/* ── Selectbox styling ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 6px !important; border-color: #E2E6EF !important;
    font-size: 0.875rem !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return {
        "diabetes": joblib.load("models/diabetes_model.pkl"),
        "heart":    joblib.load("models/heart_model.pkl"),
        "covid":    joblib.load("models/covid_model.pkl"),
    }

models = load_models()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand-name">MedPredict AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Clinical Risk Screening System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4B5A72;margin-bottom:0.6rem;">Screening Modules</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-pill">🩸&nbsp; Diabetes Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-pill">❤️&nbsp; Cardiac Mortality Risk</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-pill">🦠&nbsp; COVID-19 Risk</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#4B5A72;margin-bottom:0.5rem;">Developer</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.875rem;color:#C9D1E0;">Haseeb Tariq</div>', unsafe_allow_html=True)
    st.markdown('<div class="meta-row">BS Information Technology</div>', unsafe_allow_html=True)
    st.markdown('<div class="meta-row">Specialisation: ML / AI</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;color:#4B5A72;line-height:1.6;">Models trained on real medical datasets using Scikit-learn pipelines with StandardScaler and Random Forest classifiers.</div>', unsafe_allow_html=True)

# ─── Page Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div class="page-header-left">
    <h1>Multi-Disease Prediction System</h1>
    <p>AI-powered early risk screening · For educational use only</p>
  </div>
  <div class="badge">⚕ Clinical AI · v1.0</div>
</div>
""", unsafe_allow_html=True)

disease = st.selectbox(
    "Screening module",
    ["🩸 Diabetes", "❤️ Heart Disease", "🦠 COVID-19"],
    label_visibility="collapsed",
)

st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# ── Shared result renderer ────────────────────────────────────────────────────
def render_result(positive, prob, disease_name, high_title, low_title, high_desc, low_desc, high_precautions, low_precautions):
    # Probability block
    if prob is not None:
        color = "#DC2626" if positive else "#16A34A"
        st.markdown(f"""
        <div class="prob-block">
            <span class="prob-label">Risk Probability</span>
            <span class="prob-value" style="color:{color}">{prob*100:.1f}%</span>
            <span class="prob-sub">Based on {disease_name} prediction model</span>
        </div>""", unsafe_allow_html=True)
        st.progress(float(prob))

    if positive:
        st.markdown(f"""
        <div class="result-panel result-high">
            <div class="result-header">
                <span class="result-icon">⚠️</span>
                <span class="result-label">High Risk Detected</span>
            </div>
            <div class="result-title">{high_title}</div>
            <p class="result-desc">{high_desc}</p>
        </div>""", unsafe_allow_html=True)
        items = "".join(f"<li>{p}</li>" for p in high_precautions)
        st.markdown(f"""
        <div class="precaution-panel">
            <div class="precaution-title">Recommended Precautions</div>
            <ul class="precaution-list">{items}</ul>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-panel result-low">
            <div class="result-header">
                <span class="result-icon">✅</span>
                <span class="result-label">Low Risk</span>
            </div>
            <div class="result-title">{low_title}</div>
            <p class="result-desc">{low_desc}</p>
        </div>""", unsafe_allow_html=True)
        items = "".join(f"<li>{p}</li>" for p in low_precautions)
        st.markdown(f"""
        <div class="precaution-panel">
            <div class="precaution-title">Preventive Tips</div>
            <ul class="precaution-list">{items}</ul>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DIABETES
# ═══════════════════════════════════════════════════════════════════════════════
if disease == "🩸 Diabetes":
    st.markdown('<div class="section-card"><div class="section-card-title">Patient Information</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        age         = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, value=0, help="Enter 0 if not applicable")
        glucose     = st.slider("Glucose level (mg/dL)", 0, 250, 100, help="Normal fasting: 70–100 mg/dL")
        insulin     = st.slider("Insulin level (μU/mL)", 0, 900, 80, help="Normal fasting: 2–25 μU/mL")
    with col2:
        bmi  = st.slider("BMI", 10.0, 70.0, 25.0, step=0.1, help="Normal: 18.5–24.9 · Overweight: 25–29.9 · Obese: 30+")
        bp   = st.slider("Blood pressure (mm Hg)", 40, 180, 80, help="Normal diastolic: 60–80 mm Hg")
        skin = st.slider("Skin fold thickness (mm)", 0, 100, 20, help="Triceps skinfold. Normal ≈ 20 mm")
        dpf  = st.slider("Diabetes family history score", 0.0, 2.5, 0.5, step=0.01, help="Diabetes Pedigree Function. 0 = no family history")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run Diabetes Screening", use_container_width=True):
        input_data = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = models["diabetes"].predict(input_data)[0]
        prob = None
        try:
            prob = models["diabetes"].predict_proba(input_data)[0][1]
        except Exception:
            pass
        render_result(
            positive    = (result == 1),
            prob        = prob,
            disease_name= "Diabetes",
            high_title  = "Elevated Diabetes Risk",
            low_title   = "No Significant Diabetes Risk",
            high_desc   = "This patient's profile indicates elevated risk. Prompt consultation with a physician or endocrinologist is recommended.",
            low_desc    = "No significant risk indicators detected. Continue maintaining a healthy lifestyle and schedule routine checkups.",
            high_precautions = [
                "Consult a physician or endocrinologist promptly",
                "Reduce refined sugars and processed carbohydrates",
                "Exercise at least 30 minutes daily — brisk walking or cycling",
                "Monitor blood glucose levels regularly",
                "Maintain a healthy body weight",
                "Avoid smoking and limit alcohol consumption",
            ],
            low_precautions = [
                "Maintain a balanced diet rich in vegetables and whole grains",
                "Stay physically active — at least 150 minutes per week",
                "Schedule annual fasting blood glucose screenings",
            ],
        )

# ═══════════════════════════════════════════════════════════════════════════════
# HEART DISEASE
# ═══════════════════════════════════════════════════════════════════════════════
elif disease == "❤️ Heart Disease":
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="section-card"><div class="section-card-title">Measurements at Age 50</div>', unsafe_allow_html=True)
        age_50  = st.slider("Age at first recording", 40, 60, 50)
        md_50   = st.selectbox("Doctor visit frequency (age 50)", ["Rarely / Never", "Occasionally", "Regularly"], key="md_50")
        sbp_50  = st.slider("Systolic BP at 50 (mm Hg)", 80, 220, 120, help="Normal: <120 · High: 140+")
        dbp_50  = st.slider("Diastolic BP at 50 (mm Hg)", 40, 140, 80, help="Normal: <80 · High: 90+")
        ht_50   = st.slider("Height (cm)", 140, 210, 170)
        wt_50   = st.slider("Weight at 50 (kg)", 40, 180, 75)
        chol_50 = st.slider("Cholesterol at 50 (mg/dL)", 100, 400, 200, help="Normal: <200 · Borderline: 200–239 · High: 240+")
        ses     = st.selectbox("Socioeconomic status", ["Low", "Middle", "High"], help="Affects healthcare access and lifestyle factors")
        cl      = st.selectbox("Smoking status", ["Never Smoked", "Former Smoker", "Current Smoker"])
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-card"><div class="section-card-title">Measurements at Age 62</div>', unsafe_allow_html=True)
        md_62   = st.selectbox("Doctor visit frequency (age 62)", ["Rarely / Never", "Occasionally", "Regularly"], key="md_62")
        sbp_62  = st.slider("Systolic BP at 62 (mm Hg)", 80, 240, 130)
        dbp_62  = st.slider("Diastolic BP at 62 (mm Hg)", 40, 140, 82)
        chol_62 = st.slider("Cholesterol at 62 (mg/dL)", 100, 400, 215)
        wt_62   = st.slider("Weight at 62 (kg)", 40, 180, 78)
        st.markdown('<div class="section-card-title" style="margin-top:1rem">Diagnosis</div>', unsafe_allow_html=True)
        ihd_dx  = st.radio("Diagnosed with Ischemic Heart Disease (IHD)?", ["No", "Yes"],
                           help="IHD = reduced blood supply to the heart due to narrowed arteries", horizontal=True)
        st.markdown("</div>", unsafe_allow_html=True)

    md_map  = {"Rarely / Never": 0, "Occasionally": 1, "Regularly": 2}
    ses_map = {"Low": 1, "Middle": 2, "High": 3}
    cl_map  = {"Never Smoked": 0, "Former Smoker": 1, "Current Smoker": 2}

    if st.button("Run Cardiac Risk Screening", use_container_width=True):
        input_data = pd.DataFrame([[
            age_50, md_map[md_50], sbp_50, dbp_50, ht_50, wt_50, chol_50,
            ses_map[ses], cl_map[cl],
            md_map[md_62], sbp_62, dbp_62, chol_62, wt_62,
            1 if ihd_dx == "Yes" else 0,
        ]], columns=["AGE_50","MD_50","SBP_50","DBP_50","HT_50","WT_50","CHOL_50",
                     "SES","CL_STATUS","MD_62","SBP_62","DBP_62","CHOL_62","WT_62","IHD_DX"])
        result = models["heart"].predict(input_data)[0]
        prob = None
        try:
            prob = models["heart"].predict_proba(input_data)[0][1]
        except Exception:
            pass
        render_result(
            positive    = (result == 1),
            prob        = prob,
            disease_name= "Cardiac Mortality",
            high_title  = "Elevated Cardiac Mortality Risk",
            low_title   = "No Significant Cardiac Risk",
            high_desc   = "Longitudinal indicators suggest elevated cardiovascular risk. Immediate evaluation by a cardiologist is advised.",
            low_desc    = "No significant cardiac risk detected based on longitudinal measurements.",
            high_precautions = [
                "Seek evaluation from a cardiologist immediately",
                "Adopt a heart-healthy diet — low sodium, low saturated fat",
                "Cease smoking — significantly raises ischemic heart disease risk",
                "Monitor blood pressure and cholesterol at regular intervals",
                "Engage in moderate cardio exercise with doctor clearance",
                "Manage stress through structured relaxation techniques",
                "Maintain a healthy body weight",
            ],
            low_precautions = [
                "Annual cholesterol and blood pressure checks",
                "Follow a Mediterranean-style diet",
                "Aim for at least 150 minutes of moderate exercise per week",
            ],
        )

# ═══════════════════════════════════════════════════════════════════════════════
# COVID-19
# ═══════════════════════════════════════════════════════════════════════════════
elif disease == "🦠 COVID-19":

    # ── Exact column names as fitted ─────────────────────────────────────────
    COVID_COLUMNS = [
        "Breathing Problem", "Fever", "Dry Cough", "Sore throat",
        "Running Nose", "Asthma", "Chronic Lung Disease", "Headache",
        "Heart Disease", "Diabetes", "Hyper Tension", "Fatigue ",
        "Gastrointestinal ", "Abroad travel", "Contact with COVID Patient",
        "Attended Large Gathering", "Visited Public Exposed Places",
        "Family working in Public Exposed Places", "Wearing Masks",
        "Sanitization from Market",
    ]

    # UI key → exact model column (handles trailing spaces + casing diffs)
    DISPLAY_TO_COL = {
        "Breathing Problem":                        "Breathing Problem",
        "Fever":                                    "Fever",
        "Dry Cough":                                "Dry Cough",
        "Sore Throat":                              "Sore throat",
        "Running Nose":                             "Running Nose",
        "Headache":                                 "Headache",
        "Fatigue":                                  "Fatigue ",
        "Gastrointestinal":                         "Gastrointestinal ",
        "Asthma":                                   "Asthma",
        "Chronic Lung Disease":                     "Chronic Lung Disease",
        "Heart Disease":                            "Heart Disease",
        "Diabetes":                                 "Diabetes",
        "Hyper Tension":                            "Hyper Tension",
        "Abroad travel":                            "Abroad travel",
        "Contact with COVID Patient":               "Contact with COVID Patient",
        "Attended Large Gathering":                 "Attended Large Gathering",
        "Visited Public Exposed Places":            "Visited Public Exposed Places",
        "Family working in Public Exposed Places":  "Family working in Public Exposed Places",
        "Wearing Masks":                            "Wearing Masks",
        "Sanitization from Market":                 "Sanitization from Market",
    }

    symptom_qs = [
        ("Breathing Problem",  "Difficulty breathing?"),
        ("Fever",              "Fever above 38 °C / 100.4 °F?"),
        ("Dry Cough",          "Persistent dry cough?"),
        ("Sore Throat",        "Sore throat?"),
        ("Running Nose",       "Runny nose?"),
        ("Headache",           "Frequent headaches?"),
        ("Fatigue",            "Unusual fatigue or tiredness?"),
        ("Gastrointestinal",   "Nausea, vomiting, or diarrhoea?"),
    ]
    condition_qs = [
        ("Asthma",              "Pre-existing asthma?"),
        ("Chronic Lung Disease","Chronic lung disease?"),
        ("Heart Disease",       "Pre-existing heart condition?"),
        ("Diabetes",            "Diabetes?"),
        ("Hyper Tension",       "Hypertension?"),
    ]
    exposure_qs = [
        ("Abroad travel",                        "International travel in last 14 days?"),
        ("Contact with COVID Patient",            "Close contact with confirmed COVID-19 case?"),
        ("Attended Large Gathering",              "Attended a large gathering recently?"),
        ("Visited Public Exposed Places",         "Visited crowded public places?"),
        ("Family working in Public Exposed Places","Family member works in high-exposure setting?"),
        ("Wearing Masks",                         "Consistently wearing a mask in public?"),
        ("Sanitization from Market",              "Sanitizing items brought from outside?"),
    ]

    inputs_dict = {}
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown('<div class="section-card"><div class="section-card-title">Current Symptoms</div>', unsafe_allow_html=True)
        for key, question in symptom_qs:
            ans = st.radio(question, ["No", "Yes"], key=key, horizontal=True)
            inputs_dict[key] = 1 if ans == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-card"><div class="section-card-title">Pre-existing Conditions</div>', unsafe_allow_html=True)
        for key, question in condition_qs:
            ans = st.radio(question, ["No", "Yes"], key=key, horizontal=True)
            inputs_dict[key] = 1 if ans == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-card"><div class="section-card-title">Exposure & Safety</div>', unsafe_allow_html=True)
        for key, question in exposure_qs:
            ans = st.radio(question, ["No", "Yes"], key=key, horizontal=True)
            inputs_dict[key] = 1 if ans == "Yes" else 0
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Run COVID-19 Screening", use_container_width=True):
        # Build row with exact column names, integer values
        row = {DISPLAY_TO_COL[k]: inputs_dict[k] for k in inputs_dict}
        input_data = pd.DataFrame([row], columns=COVID_COLUMNS)
        result = models["covid"].predict(input_data)[0]

        prob = None
        try:
            classes = list(models["covid"].classes_)
            # classes may be [0,1] or ["No","Yes"] depending on how model was saved
            pos_label = 1 if 1 in classes else "Yes"
            prob = models["covid"].predict_proba(input_data)[0][classes.index(pos_label)]
        except Exception:
            pass

        positive = (result == 1 or result == "Yes")
        render_result(
            positive    = positive,
            prob        = prob,
            disease_name= "COVID-19",
            high_title  = "Elevated COVID-19 Risk",
            low_title   = "No Significant COVID-19 Risk",
            high_desc   = "This patient's symptom and exposure profile is consistent with COVID-19 risk. Isolation and testing are strongly advised.",
            high_precautions = [
                "Self-isolate immediately and avoid contact with others",
                "Get a PCR or rapid antigen test as soon as possible",
                "Wear a well-fitting N95 or surgical mask",
                "Monitor oxygen levels — SpO₂ should stay above 94%",
                "Stay hydrated and rest adequately",
                "Contact a healthcare provider or COVID helpline",
                "Seek emergency care if breathing becomes severely difficult",
            ],
            low_desc    = "No strong risk indicators detected. Continue following standard health precautions.",
            low_precautions = [
                "Continue wearing a mask in crowded or indoor spaces",
                "Wash hands frequently for at least 20 seconds",
                "Maintain physical distance in high-risk environments",
                "Stay up to date with COVID-19 vaccinations",
            ],
        )

# ─── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    <span>⚕️</span>
    <span><strong>Medical Disclaimer:</strong> This application is developed for educational and academic purposes only.
    It does not constitute medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare professional for any medical concern.</span>
</div>
""", unsafe_allow_html=True)