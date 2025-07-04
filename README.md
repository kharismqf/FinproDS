# 🧠 Income Classification with XGBoost

This project develops a machine learning model that predicts whether an individual's income is more or less than $50K per year using demographic and occupational data.

## 🚀 App Overview

The app is built using **Streamlit** and includes:
- Exploratory Data Analysis (EDA)
- Interactive model prediction form
- Visual dashboards
- Business insights
- Feature importance analysis
- About the creator

## 📊 Dataset
- Source: UCI Adult Dataset (modified)
- Rows: ~49,000
- Target: `income` (`<=50K` or `>50K`)

## 🧠 Model
- Final model: `XGBoostClassifier`
- Tuned using: `RandomizedSearchCV`
- Performance:
    - Accuracy: 87.2%
    - F1-Score: 0.707
    - ROC-AUC: 0.927

## 📁 Project Structure

Finpro/
│
├── dataset/ # Contains train & test data
├── images/ # Contains images for dashboard & explanation
├── models/ # Saved pipeline models (e.g., income_xgb.pkl)
├── model/ # Python modules for pages (trythemodel.py, aboutme.py, etc.)
├── .streamlit/ # Streamlit config
│ └── config.toml
├── streamlit_app.py # Main Streamlit app
├── README.md # Project readme file
└── requirements.txt # Python dependencies

