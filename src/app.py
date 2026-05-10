import streamlit as st
import numpy as np
import joblib
st.title("Heart Disease Risk Prediction System")
st.write("This application predicts the risk level of heart disease based on patient health parameters.")
model = joblib.load("../models/logistic_regression_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.subheader("Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=120, value=50, key="age")
sex = st.selectbox("Sex", [0, 1], key="sex")  # 0 = Female, 1 = Male
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4], key="cp")
bp = st.number_input("Resting Blood Pressure", value=120, key="bp")
chol = st.number_input("Cholesterol", value=200, key="chol")
fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1], key="fbs")
ekg = st.selectbox("EKG Results", [0, 1, 2], key="ekg")
max_hr = st.number_input("Max Heart Rate", value=150, key="max_hr")
ex_angina = st.selectbox("Exercise Induced Angina", [0, 1], key="ex_angina")
st_depression = st.number_input("ST Depression", value=1.0, key="st_depression")
slope = st.selectbox("Slope of ST", [1, 2, 3], key="slope")
vessels = st.selectbox("Number of Vessels (Fluoroscopy)", [0, 1, 2, 3], key="vessels")
thal = st.selectbox("Thallium Test Result", [3, 6, 7], key="thal")

st.subheader("Prediction")

if st.button("Predict Risk"):
    input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg,
                             max_hr, ex_angina, st_depression,
                             slope, vessels, thal]])

    input_scaled = scaler.transform(input_data)
    probability = model.predict_proba(input_scaled)[0][1]

    if probability < 0.3:
        risk = "Low Risk"
    elif probability < 0.6:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.success(f"Risk Level: {risk}")
    st.write(f"Probability of Heart Disease: {probability:.2f}")