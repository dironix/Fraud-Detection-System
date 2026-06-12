import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_model.pkl"
)

model = joblib.load(model_path)

st.title("Credit Card Fraud Detection")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)

    data["Prediction"] = predictions

    st.write(data.head())

    frauds = (predictions == 1).sum()

    st.success(
        f"Fraudulent Transactions Detected: {frauds}"
    )