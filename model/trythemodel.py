# model/trythemodel.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

@st.cache_resource
def load_pipe():
    return joblib.load(Path("models/income_xgb_tuned.pkl"))

pipe = load_pipe()

# Education mapping
EDU2NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8,
    "HS-grad": 9, "Some-college": 10, "Assoc-acdm": 11, "Assoc-voc": 11,
    "Bachelors": 13, "Masters": 15, "Prof-school": 15, "Doctorate": 16
}

def trythemodel():
    st.subheader("🚀 Try the Income Classification Model")

    with st.form("user_inputs"):
        st.markdown("### 📥 Enter Personal & Work Information")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("🎂 Age", 17, 90, 30)
            capital_gain = st.number_input("📈 Capital Gain", 0, 100_000, 0, step=500)
            capital_loss = st.number_input("📉 Capital Loss", 0, 50_000, 0, step=500)
            hours_per_week = st.slider("⏰ Hours per Week", 1, 99, 40)
            education = st.selectbox("🎓 Education", list(EDU2NUM.keys()))
            marital_status = st.selectbox("💍 Marital Status", [
                "Married-civ-spouse", "Divorced", "Never-married",
                "Separated", "Widowed", "Married-spouse-absent"
            ])

        with col2:
            workclass = st.selectbox("🏢 Workclass", [
                "Private", "Self-emp-not-inc", "Local-gov", "State-gov",
                "Self-emp-inc", "Federal-gov", "Without-pay", "Never-worked"
            ])
            occupation_grouped = st.selectbox("💼 Occupation Grouped", [
                "Professional", "Executive", "Clerical", "Sales",
                "Skilled Labor", "Unskilled Labor", "Service", "Military"
            ])
            relationship = st.selectbox("👪 Relationship", [
                "Husband", "Wife", "Own-child", "Not-in-family",
                "Other-relative", "Unmarried"
            ])
            race = st.selectbox("🌎 Race", [
                "White", "Black", "Asian-Pac-Islander",
                "Amer-Indian-Eskimo", "Other"
            ])
            sex = st.radio("⚧️ Sex", ["Male", "Female"], horizontal=True)
            native_region = st.selectbox("🗺️ Native Region", [
                "United-States", "Latin America", "Caribbean",
                "Asia", "Europe", "Canada", "Other"
            ])

        submitted = st.form_submit_button("🎯 Predict")

    if submitted:
        education_num = EDU2NUM[education]

        # ✅ Using your original structure
        X_new = pd.DataFrame({
            "age": [age],
            "workclass": [workclass],
            "education-num": [education_num],
            "marital-status": [marital_status],
            "occupation_grouped": [occupation_grouped],
            "relationship": [relationship],
            "race": [race],
            "sex": [sex],
            "capital-gain": [capital_gain],
            "capital-loss": [capital_loss],
            "hours-per-week": [hours_per_week],
            "native_region": [native_region],
        })

        # Match pipeline columns
        expected_cols = list(pipe.named_steps["pre"].feature_names_in_)
        X_new = X_new[expected_cols]

        with st.spinner("🔍 Making prediction..."):
            proba = pipe.predict_proba(X_new)[0, 1]
            pred = pipe.predict(X_new)[0]

        # 🧾 Prediction result
        if pred == 1:
            st.success("💰 **Prediction: > 50K**")
        else:
            st.info("📦 **Prediction: ≤ 50K**")

        # Show input summary
        st.markdown("### 🔎 Input Summary")
        st.dataframe(X_new)
        st.caption(f"🎯 Prediction Label: {pred}")
