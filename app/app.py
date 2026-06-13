import streamlit as st
import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_model.pkl"
)

model = joblib.load(model_path)

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection System")

st.markdown(
    """
    Upload transaction data and detect
    fraudulent transactions using
    an XGBoost Machine Learning model.
    """
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    predictions = model.predict(data)

    data["Prediction"] = predictions

    st.subheader(
    "Prediction Results"
)

    st.dataframe(
    data.head(20)
)

    frauds = (predictions == 1).sum()

    st.metric(
    "Fraudulent Transactions",
    frauds
)