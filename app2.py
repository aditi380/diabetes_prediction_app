import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

# 🧠 Load model and scaler
scaler = joblib.load("scaler.pkl")
model = xgb.XGBClassifier()
model.load_model("diabetes_xgb_model.json")

# 🌐 Streamlit page setup
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="💉", layout="wide")

# 💅 Custom CSS for modern, elegant design
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #f8fbff, #e6f2ff);
        font-family: 'Segoe UI';
    }
    h1 {
        color: #004e7c;
        text-align: center;
        font-size: 42px !important;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #004e7c;
        color: white;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 18px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0078a8;
        transform: scale(1.02);
    }
    .card {
        background-color: #ecf6ff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, #004e7c, #00aaff);
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# 🏥 Header Section
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=120)
st.markdown("<h1>💉 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;color:#333;'>AI-powered tool for early detection and prevention of diabetes risk.</p>", unsafe_allow_html=True)
st.write("<hr>", unsafe_allow_html=True)

# 🧾 Input Fields
st.subheader("🩺 Enter Patient Medical Details")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
        skin = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    with col2:
        insulin = st.number_input("Insulin Level (IU/mL)", min_value=0.0, max_value=900.0, value=80.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

# 🧩 Feature Transformation
bmi_glucose = bmi * glucose
age_insulin = age * insulin
features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age, bmi_glucose, age_insulin]])
scaled_features = scaler.transform(features)

# 🔍 Prediction Button
st.write("<hr>", unsafe_allow_html=True)
if st.button("Predict Diabetes"):
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        st.error("⚠️ The person is **likely to have Diabetes.**")
        st.markdown("<p style='text-align:center;color:red;font-size:20px;'>Early diagnosis is crucial — please consult a healthcare professional.</p>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3448/3448336.png", width=180)
    else:
        st.success("✅ The person is **NOT Diabetic.**")
        st.markdown("<p style='text-align:center;color:green;font-size:20px;'>Healthy blood sugar level detected. Maintain a balanced lifestyle!</p>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=180)

st.write("<hr>", unsafe_allow_html=True)

# 📊 Footer (no personal name)
st.markdown("""
    <div style='text-align:center; color:#555;'>
        <p style='font-size:15px;'>
        Developed as part of an Academic Project 🧠 | 
        Built using <b>Python, Streamlit, XGBoost & Scikit-learn</b>
        </p>
    </div>
""", unsafe_allow_html=True)
