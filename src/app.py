import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns


model = joblib.load("../models/diabetes_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

st.set_page_config(page_title="Explainable AI - Diabetes Prediction", layout="centered")
st.title("Explainable AI - Diabetes Prediction")


st.subheader("Enter Patient Details")
pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
glucose = st.number_input("Glucose", min_value=0, step=1)
bp = st.number_input("Blood Pressure", min_value=0, step=1)
skin = st.number_input("Skin Thickness", min_value=0, step=1)
insulin = st.number_input("Insulin", min_value=0, step=1)
bmi = st.number_input("BMI", min_value=0.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)
age = st.number_input("Age", min_value=0, step=1)

input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
input_scaled = scaler.transform(input_data)

feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DPF","Age"]


prediction = model.predict(input_scaled)[0]
probability = model.predict_proba(input_scaled)[0][prediction]

if prediction == 1:
    st.subheader("Prediction: Diabetic")
    st.write(f"The patient **is diabetic** with probability {probability*100:.2f}%.")
else:
    st.subheader("Prediction: Not Diabetic")
    st.write(f"The patient **is not diabetic** with probability {probability*100:.2f}%.")


st.subheader("Feature Contributions")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)


shap_contributions = shap_values[0]
colors = ['#FF4136' if val>0 else '#2ECC40' for val in shap_contributions]  # red=high risk, green=low
fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(x=shap_contributions, y=feature_names, palette=colors)
ax.set_xlabel("SHAP Value (Impact on Prediction)")
st.pyplot(fig)
plt.clf()


st.subheader("Explanation")
for i, feat in enumerate(feature_names):
    impact = "increases risk" if shap_contributions[i] > 0 else "decreases risk"
    st.write(f"- **{feat}**: {impact} (SHAP value: {shap_contributions[i]:.3f})")


st.subheader("Risk Reduction Advice")

advice = []

if shap_contributions[1] > 0:  # Glucose
    advice.append("- Consider monitoring and controlling your **blood glucose levels** through diet and exercise.")
if shap_contributions[5] > 0:  # BMI
    advice.append("- Maintain a healthy **BMI** through regular exercise and balanced diet.")
if shap_contributions[7] > 0:  # Age
    advice.append("- Regular check-ups are recommended due to **age-related risk**.")
if shap_contributions[2] > 0:  # Blood Pressure
    advice.append("- Keep your **blood pressure** under control with lifestyle changes or medication if needed.")
if shap_contributions[4] > 0:  # Insulin
    advice.append("- Consult your doctor regarding **insulin levels** and related health monitoring.")

if len(advice) == 0:
    st.write("All risk factors are low. Keep maintaining a healthy lifestyle!")
else:
    for item in advice:
        st.write(item)
