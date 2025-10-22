import streamlit as st
import numpy as np
import joblib
import xgboost as xgb

# 🧠 Load the trained model and scaler
scaler = joblib.load("scaler.pkl")
model = xgb.XGBClassifier()
model.load_model("diabetes_xgb_model.json")

# 🌐 Streamlit page settings
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

# 💉 Custom CSS for modern look
st.markdown("""
    <style>
    .main {
        background-color: #F7FBFF;
    }
    h1 {
        color: #0078A8;
        text-align: center;
        font-size: 38px !important;
    }
    .stButton>button {
        background-color: #0078A8;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #005f7a;
        color: white;
    }
    .card {
        background-color: #E8F6FB;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# 🏥 App header with logo
col1, col2, col3 = st.columns([1,3,1])
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=120)
st.markdown("<h1>🩺 Diabetes Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>AI-powered system to assist doctors in early diabetes detection.</p>", unsafe_allow_html=True)
st.write("---")

# 🧾 Input Section
st.subheader("👩‍⚕️ Enter Patient Medical Details")

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

# 🧩 Feature Engineering (same as training)
bmi_glucose = bmi * glucose
age_insulin = age * insulin
features = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age, bmi_glucose, age_insulin]])
scaled_features = scaler.transform(features)

# 🩺 Prediction Button
st.write("---")
if st.button("🔍 Predict Diabetes"):
    prediction = model.predict(scaled_features)[0]

    if prediction == 1:
        st.error("⚠️ The person is **likely to have Diabetes.**")
        st.markdown("<p style='text-align:center;color:red;font-size:20px;'>Early diagnosis is crucial — please consult a doctor.</p>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/3448/3448336.png", width=180)
    else:
        st.success("✅ The person is **NOT Diabetic.**")
        st.markdown("<p style='text-align:center;color:green;font-size:20px;'>Healthy blood sugar level detected.</p>", unsafe_allow_html=True)
        st.image("https://cdn-icons-png.flaticon.com/512/2966/2966484.png", width=180)

st.write("---")

# 📊 Footer
st.markdown("""
    <div style='text-align:center;'>
    <p style='font-size:16px;'>Developed by <b>Aditi Jain</b> | 3rd Year AIML 🧠<br>
    Using <b>Python, XGBoost, Scikit-learn, Streamlit</b></p>
    </div>
""", unsafe_allow_html=True)
