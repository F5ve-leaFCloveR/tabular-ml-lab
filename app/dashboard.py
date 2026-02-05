from __future__ import annotations

import streamlit as st

from tabular_ml_lab.infer import load_artifacts, predict_proba

st.set_page_config(page_title="Tabular ML Lab", layout="centered")

st.title("Income Prediction Demo")
st.write("Predict whether income exceeds $50K/year using a trained PyTorch model.")

model, preprocessor, metadata = load_artifacts("models")

with st.form("predict_form"):
    st.subheader("Personal & Work Profile")
    age = st.number_input("Age", min_value=17, max_value=90, value=38)
    workclass = st.selectbox(
        "Workclass",
        [
            "Private",
            "Self-emp-not-inc",
            "Self-emp-inc",
            "Federal-gov",
            "Local-gov",
            "State-gov",
            "Without-pay",
            "Never-worked",
        ],
    )
    fnlwgt = st.number_input("FNLWGT", min_value=10000, max_value=1500000, value=120000)
    education = st.selectbox(
        "Education",
        [
            "HS-grad",
            "Some-college",
            "Bachelors",
            "Masters",
            "Doctorate",
            "Assoc-voc",
            "Assoc-acdm",
            "11th",
            "10th",
            "9th",
            "7th-8th",
            "12th",
            "1st-4th",
            "5th-6th",
            "Preschool",
        ],
    )
    education_num = st.number_input("Education Num", min_value=1, max_value=16, value=10)
    marital_status = st.selectbox(
        "Marital Status",
        [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Separated",
            "Widowed",
            "Married-spouse-absent",
        ],
    )
    occupation = st.selectbox(
        "Occupation",
        [
            "Tech-support",
            "Craft-repair",
            "Other-service",
            "Sales",
            "Exec-managerial",
            "Prof-specialty",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Adm-clerical",
            "Farming-fishing",
            "Transport-moving",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
        ],
    )
    relationship = st.selectbox(
        "Relationship",
        [
            "Not-in-family",
            "Husband",
            "Wife",
            "Own-child",
            "Unmarried",
            "Other-relative",
        ],
    )
    race = st.selectbox(
        "Race",
        ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
    )
    sex = st.selectbox("Sex", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0)
    hours_per_week = st.slider("Hours per week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox(
        "Native Country",
        [
            "United-States",
            "Mexico",
            "Philippines",
            "Germany",
            "Canada",
            "India",
            "England",
            "China",
            "Italy",
            "Other",
        ],
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    record = {
        "age": age,
        "workclass": workclass,
        "fnlwgt": fnlwgt,
        "education": education,
        "education-num": education_num,
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": capital_gain,
        "capital-loss": capital_loss,
        "hours-per-week": hours_per_week,
        "native-country": native_country,
    }
    prob = predict_proba([record], model, preprocessor)[0]
    st.metric("Probability > $50K", f"{prob:.2%}")
