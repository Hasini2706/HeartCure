# HeartCare — Heart Disease Prediction System

A clinical-grade machine learning web application built with **Streamlit** and **Support Vector Machine (SVM)** to predict the likelihood of heart disease from cardiovascular health indicators.

---

## 📌 Features

- **Interactive Diagnostic Portal:** Enter key clinical parameters including Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, ST Depression, Slope, Number of Major Vessels, and Thalassemia.
- **Support Vector Machine (SVM) Classifier:** Utilizes an RBF-kernel SVM tuned with cross-validated grid search for robust accuracy and precision.
- **Evidence-Based Risk Categorization:** Generates categorized clinical risk indicators (Safe / Low Risk, Moderate Risk, High Risk) accompanied by probability scores and feature breakdowns.
- **Modern Medical Dashboard:** Built with custom clinical styling, visual risk meters, and responsive layouts.
- **Quick Preset Testing:** Test healthy vs. high-risk patient profiles with one-click presets.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** [Streamlit](https://streamlit.io/)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (Support Vector Classifier - SVC, StandardScaler, GridSearchCV)
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Model Serialization:** [Joblib](https://joblib.readthedocs.io/)
- **Visualization:** Matplotlib

---

## 📂 Project Structure

```text
HeartCure/
├── .streamlit/
│   └── config.toml          # Streamlit theme & UI configurations
├── assets/                  # Application visual assets & medical illustrations
│   ├── cardiac_ai_blended.png
│   ├── cardiac_ai_hero.png
│   ├── doctor.jpg
│   └── heart_3d.jpg
├── app.py                   # Main Streamlit web application
├── model.py                 # ML model training, tuning & evaluation script
├── Heart_disease_dataset.csv# Cardiovascular clinical dataset
├── svm_model.pkl            # Pre-trained tuned SVM model
├── scaler.pkl               # Fitted standard scaler
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignored files (docs, reports, caches)
└── README.md                # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Hasini2706/HeartCure.git
cd HeartCure
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to view and interact with the application.

---

## 📊 Retraining the Model

To retrain the SVM classifier on the dataset and regenerate the artifacts (`svm_model.pkl` and `scaler.pkl`):

```bash
python model.py
```

---

## ⚠️ Medical Disclaimer

This application is developed for educational and clinical decision-support research purposes. It is not intended to replace professional medical diagnosis, advice, or treatment. Always consult qualified healthcare professionals for medical evaluations.
