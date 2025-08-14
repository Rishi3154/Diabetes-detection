# 🩺 Explainable AI for Diabetes Detection

## 📌 Overview
This project implements a **Machine Learning model** that predicts whether a patient is diabetic or not, while also providing **human-understandable explanations** for its predictions using **SHAP (SHapley Additive exPlanations)**.  
The goal is to make AI decisions **transparent, interpretable, and actionable**, ensuring medical professionals and patients can trust and understand the results.

## 🚀 Features
- **Diabetes Prediction** using XGBoost  
- **Explainable AI** with SHAP visualizations  
- **Interactive Web App** built with Streamlit  
- **Risk Contribution Analysis** — Shows how each health parameter contributes to the decision  
- **Personalized Recommendations** to reduce diabetes risk based on patient data  
- **User-friendly interface** for both medical experts and patients  

## 🧠 How It Works
1. **Data Input** — User enters medical details (Glucose, BMI, Age, Blood Pressure, etc.)  
2. **Prediction** — The ML model predicts the likelihood of diabetes  
3. **Explanation** — SHAP explains each feature’s impact on the prediction  
4. **Recommendation** — Provides actionable health advice tailored to the patient  

## 🛠️ Tech Stack
- **Python** (pandas, numpy, matplotlib, seaborn)
- **Machine Learning**: XGBoost, scikit-learn
- **Explainable AI**: SHAP
- **Web App**: Streamlit

## 📂 Project Structure
📦 explainable-diabetes-ai
┣ 📂 data # Dataset(s)
┣ 📂 models # Saved trained models (.pkl)
┣ 📂 notebook # Jupyter notebook for EDA & training
┣ 📂 src # Source code for training & app
┣ 📜 app.py # Main Streamlit app
┣ 📜 req.txt # Python dependencies
┣ 📜 README.md # Project documentation

## 📊 Example Output
### 🔹 Prediction
**"The patient is NOT diabetic"** or **"The patient is diabetic"**

### 🔹 SHAP Feature Contribution
![SHAP Summary Plot](images/shap_summary.png)

### 🔹 Personalized Recommendations
- **If High BMI:** Reduce weight through diet & exercise  
- **If High Glucose:** Limit sugar intake & monitor blood sugar regularly  
- **If Low Physical Activity:** Increase daily exercise  

## 📥 Installation & Usage
```bash
# Clone the repository
git clone https://github.com/yourusername/explainable-diabetes-ai.git
cd explainable-diabetes-ai

# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app.py