import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Heart Risk Checker", layout="centered")

st.title("❤️ Heart Disease Risk Prediction App")
st.markdown("Check your heart health status using medical details. Useful for patients and doctors.")

# Sample data to train the model
def load_training_data():
    return pd.DataFrame({
        'age': [63, 37, 41, 56, 57, 45, 54, 52, 61, 60],
        'sex': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'cp': [3, 2, 1, 1, 0, 2, 1, 3, 0, 2],
        'trestbps': [145, 130, 130, 120, 120, 140, 132, 128, 138, 135],
        'chol': [233, 250, 204, 236, 354, 210, 270, 294, 230, 250],
        'fbs': [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        'restecg': [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
        'thalach': [150, 187, 172, 178, 163, 160, 165, 155, 148, 170],
        'exang': [0, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        'oldpeak': [2.3, 3.5, 1.4, 0.8, 0.6, 1.2, 2.5, 1.3, 1.9, 0.7],
        'slope': [0, 0, 2, 2, 2, 1, 1, 2, 0, 2],
        'ca': [0, 0, 0, 0, 0, 1, 2, 1, 1, 0],
        'thal': [1, 2, 2, 2, 2, 3, 0, 1, 2, 3],
        'target': [1, 1, 1, 1, 0, 1, 0, 0, 0, 1]
    })

# Sidebar form
st.sidebar.header("📝 Enter Patient Medical Details")

def get_user_input():
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

user_input = get_user_input()

# Load training data and train model
data = load_training_data()
X = data.drop("target", axis=1)
y = data["target"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict
prediction = model.predict(user_input)
proba = model.predict_proba(user_input)

# Output section
st.subheader("🔍 Patient Health Condition:")
if prediction[0] == 1:
    st.error("💔 The patient is **at risk** of heart disease.")
else:
    st.success("✅ The patient is **not at risk** of heart disease.")

st.subheader("📊 Prediction Confidence:")
st.write(f"Probability of Not at Risk: **{proba[0][0]:.2f}**")
st.write(f"Probability of At Risk: **{proba[0][1]:.2f}**")

st.markdown("---")
st.caption("This app is a demo model and should not replace professional medical advice.")
