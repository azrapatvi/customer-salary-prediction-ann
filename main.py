import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import json

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Oracle",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg:        #0a0c14;
    --surface:   #12151f;
    --card:      #181c2a;
    --border:    #252a3a;
    --accent:    #6c63ff;
    --accent2:   #00d4aa;
    --text:      #e8eaf2;
    --muted:     #6b7280;
    --danger:    #ff4d6d;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3rem 4rem !important;
    max-width: 1100px !important;
}

/* ── Hero Banner ── */
.hero {
    background: linear-gradient(135deg, #1a1040 0%, #0d1a2e 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 3rem 3.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(108,99,255,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(0,212,170,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-tag {
    display: inline-block;
    background: rgba(108,99,255,0.15);
    border: 1px solid rgba(108,99,255,0.4);
    color: #a09bff;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.3rem 0.9rem;
    border-radius: 50px;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    line-height: 1.1 !important;
    letter-spacing: -0.02em;
    margin: 0 0 0.6rem !important;
    background: linear-gradient(135deg, #ffffff 30%, #a09bff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero p {
    color: var(--muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
    max-width: 520px;
}

/* ── Section Label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card ── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(108,99,255,0.35); }

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #9ba3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.3rem !important;
}

div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6c63ff 0%, #8b5cf6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 24px rgba(108,99,255,0.4) !important;
    width: 100% !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(108,99,255,0.55) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Success result box ── */
.result-box {
    background: linear-gradient(135deg, rgba(0,212,170,0.08) 0%, rgba(108,99,255,0.08) 100%);
    border: 1px solid rgba(0,212,170,0.35);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: pop 0.35s cubic-bezier(0.34,1.56,0.64,1);
}
@keyframes pop {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}
.result-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent2);
    margin-bottom: 0.4rem;
}
.result-value {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4aa 0%, #6c63ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}

/* ── Stat chips ── */
.chip-row {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
}
.chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.4rem 0.85rem;
    font-size: 0.78rem;
    color: var(--muted);
    font-weight: 500;
}
.chip span { color: var(--text); font-weight: 600; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Toggle style for Yes/No selects ── */
div[data-testid="stSelectbox"] svg { color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">💎 ML-Powered Analytics</div>
    <h1>Salary Oracle</h1>
    <p>Fill in the customer profile below and let the neural network estimate the expected salary in seconds.</p>
</div>
""", unsafe_allow_html=True)


# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Customer Profile</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        creditscore = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
        geography   = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender      = st.selectbox("Gender", ["Male", "Female"])

    with col2:
        age            = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure         = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
        balance        = st.number_input("Balance (€)", min_value=0.0, value=0.0, format="%.2f")

    with col3:
        numofproducts  = st.number_input("No. of Products", min_value=1, max_value=4, value=1)
        hascrcard      = st.selectbox("Has Credit Card", options=[1, 0], format_func=lambda x: "Yes" if x else "No")
        isactivemember = st.selectbox("Active Member", options=[1, 0], format_func=lambda x: "Yes" if x else "No")

    st.markdown('</div>', unsafe_allow_html=True)


# ── Live Summary Chips ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="chip-row">
  <div class="chip">Score <span>{creditscore}</span></div>
  <div class="chip">Age <span>{age}</span></div>
  <div class="chip">Region <span>{geography}</span></div>
  <div class="chip">Tenure <span>{tenure} yr</span></div>
  <div class="chip">Balance <span>€{balance:,.0f}</span></div>
  <div class="chip">Products <span>{numofproducts}</span></div>
  <div class="chip">Credit Card <span>{"✓" if hascrcard else "✗"}</span></div>
  <div class="chip">Active <span>{"✓" if isactivemember else "✗"}</span></div>
</div>
""", unsafe_allow_html=True)


# ── Raw Data Preview ───────────────────────────────────────────────────────────
with st.expander("🔍 Preview Raw Input Data"):
    input_data = pd.DataFrame([{
        "creditscore": creditscore, "geography": geography, "gender": gender,
        "age": age, "tenure": tenure, "balance": balance,
        "numofproducts": numofproducts, "hascrcard": hascrcard,
        "isactivemember": isactivemember
    }])
    st.dataframe(input_data, use_container_width=True, hide_index=True)


# ── Prediction Button ──────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("⚡  Run Prediction")

if predict_clicked:
    with st.spinner("Crunching numbers…"):
        try:
            with open('models/gender_encoding.json', 'r') as f:
                gender_encoding = json.load(f)
            preprocessor = joblib.load("models/preprocessor.joblib")
            model        = load_model("models/model.keras")

            df = input_data.copy()
            df['gender'] = df['gender'].map(gender_encoding)
            X_processed  = preprocessor.transform(df)
            prediction   = model.predict(X_processed)
            salary        = prediction[0][0] if hasattr(prediction[0], '__len__') else prediction[0]

            st.markdown(f"""
            <div class="result-box">
                <div class="result-label">Predicted Annual Salary</div>
                <div class="result-value">€ {salary:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(f"Model file not found: {e}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color: #3a3f52; font-size: 0.78rem; letter-spacing: 0.06em;">
    SALARY ORACLE &nbsp;·&nbsp; Neural Network Inference &nbsp;·&nbsp; Internal Tool
</div>
""", unsafe_allow_html=True)