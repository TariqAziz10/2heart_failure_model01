import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Cache the model loading for performance
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load("best_heart_failure_xgb_model.pkl")
    return model

model = load_model()

st.title("Heart Failure Prediction Dashboard")
st.write("Enter patient clinical details below to predict the risk of death event.")

# Create input widgets for each feature (order must match the model training)
age = st.number_input("Age", min_value=1, max_value=120, value=60, step=1)
anaemia = st.selectbox("Anaemia (0: No, 1: Yes)", options=[0, 1], index=0)
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0, value=100, step=1)
diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", options=[0, 1], index=0)
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, value=35, step=1)
high_blood_pressure = st.selectbox("High Blood Pressure (0: No, 1: Yes)", options=[0, 1], index=0)
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, value=200000.0, step=1000.0, format="%.2f")
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, value=1.0, step=0.1, format="%.2f")
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, value=137, step=1)
sex = st.selectbox("Sex (0: Female, 1: Male)", options=[0, 1], index=0)
smoking = st.selectbox("Smoking (0: No, 1: Yes)", options=[0, 1], index=0)
time = st.number_input("Follow-up Time (days)", min_value=0, value=100, step=1)

if st.button("Predict"):
    # Construct the input data array; order should match training
    input_data = np.array([[age,
                            anaemia,
                            creatinine_phosphokinase,
                            diabetes,
                            ejection_fraction,
                            high_blood_pressure,
                            platelets,
                            serum_creatinine,
                            serum_sodium,
                            sex,
                            smoking,
                            time]])
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[:, 1]

    # Map prediction result to a human-readable output
    result = "High risk of death event" if prediction[0] == 1 else "Low risk of death event"

    st.subheader("Prediction Result")
    st.write("**Outcome:**", result)
    st.write("**Probability of Death Event:** {:.2f}%".format(prediction_proba[0] * 100))
