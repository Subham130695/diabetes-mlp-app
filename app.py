import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

model = keras.models.load_model("diabetes_mlp.keras")
scaler = joblib.load("diabetes_scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")
impute_medians = joblib.load("impute_medians.pkl")

st.title("🩺 Diabetes Prediction App (MLP)")
st.write("Enter patient details to predict diabetes likelihood.")

pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=250, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict"):
    if glucose == 0 or blood_pressure == 0 or bmi == 0:
        st.warning("Glucose, Blood Pressure, and BMI cannot be 0. Please enter valid values.")
    else:
        # Replicate training-time imputation: treat 0 as missing for these fields too
        glucose_v = glucose
        bp_v = blood_pressure
        bmi_v = bmi
        skin_v = skin_thickness if skin_thickness != 0 else impute_medians['SkinThickness']
        insulin_v = insulin if insulin != 0 else impute_medians['Insulin']

        input_dict = {
            'Pregnancies': pregnancies, 'Glucose': glucose_v, 'BloodPressure': bp_v,
            'SkinThickness': skin_v, 'Insulin': insulin_v, 'BMI': bmi_v,
            'DiabetesPedigreeFunction': dpf, 'Age': age
        }
        df_input = pd.DataFrame([input_dict])

        age_group = pd.cut(df_input['Age'], bins=[20,30,40,50,60,90], labels=['21-30','31-40','41-50','51-60','60+'])
        bmi_cat = 'Underweight' if bmi_v < 18.5 else 'Normal' if bmi_v < 25 else 'Overweight' if bmi_v < 30 else 'Obese'
        gluc_cat = 'Normal' if glucose_v < 100 else 'Prediabetic' if glucose_v < 126 else 'Diabetic_Range'

        df_input['Glucose_BMI_Interaction'] = glucose_v * bmi_v
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
        input_scaled = scaler.transform(df_input)
        prob = model.predict(input_scaled)[0][0]
        prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

        st.subheader(f"Prediction: {prediction}")
        st.write(f"Probability: {prob*100:.2f}%")
