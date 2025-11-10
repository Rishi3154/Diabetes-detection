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
health/
├─ src/
│  └─ app.py                 # Streamlit UI
├─ models/
│  ├─ diabetes_model.pkl     # pretrained model
│  └─ scaler.pkl             # preprocessing scaler
├─ data/
│  └─ diabetes.csv           # reference dataset (ignored in git ideally)
├─ notebooks/
│  └─ model_exploration.ipynb
├─ screenshots/
│  └─ shap_summary.png
├─ requirements.txt
├─ README.md
└─ .gitignore

## 📊 Example Output
### 🔹 Prediction
**"The patient is NOT diabetic"** or **"The patient is diabetic"**

### 🔹 SHAP Feature Contribution
![SHAP Summary Plot](screenshots/shap_summary.png)

### 🔹 Personalized Recommendations
- **If High BMI:** Reduce weight through diet & exercise  
- **If High Glucose:** Limit sugar intake & monitor blood sugar regularly  
- **If Low Physical Activity:** Increase daily exercise  

## 📥 Installation & Usage

## 🚀 Quickstart

```bash
# 1) Clone
git clone https://github.com/Rishi3154/Diabetes-detection.git
cd Diabetes-detection

# 2) Create env (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install
pip install --upgrade pip
pip install -r requirements.txt

# 4) Run the app
streamlit run src/app.py
