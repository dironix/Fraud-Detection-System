import streamlit as st
import pandas as pd
import joblib
import os

# =========================
# PATHS
# =========================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Fraud Detection System")

st.sidebar.markdown(
    "[GitHub Repository](https://github.com/dironix/Fraud-Detection-System)"
)

# Model Selection
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["XGBoost", "Random Forest"]
)

if model_choice == "XGBoost":
    model_path = os.path.join(
        BASE_DIR,
        "models",
        "xgboost_model.pkl"
    )
else:
    model_path = os.path.join(
        BASE_DIR,
        "models",
        "random_forest_model.pkl"
    )

model = joblib.load(model_path)

# =========================
# MODEL PERFORMANCE
# =========================

st.sidebar.header("Model Performance")

st.sidebar.metric(
    "ROC-AUC",
    "0.9831"
)

st.sidebar.metric(
    "Precision",
    "0.79"
)

st.sidebar.metric(
    "Recall",
    "0.85"
)

st.sidebar.metric(
    "F1 Score",
    "0.82"
)

# =========================
# FEATURE IMPORTANCE
# =========================

feature_importance = pd.DataFrame(
    {
        "Feature": [
            "V14", "V4", "V12", "V17",
            "V3", "V1", "V8", "V13",
            "V10", "V7"
        ],
        "Importance": [
            0.702731,
            0.056232,
            0.030155,
            0.020600,
            0.015473,
            0.013341,
            0.013259,
            0.012075,
            0.009359,
            0.009217
        ]
    }
)

# =========================
# MAIN PAGE
# =========================

st.title("💳 Credit Card Fraud Detection System")

st.write(
    """
    Upload a CSV file containing credit card transactions.
    
    Required columns:
    Time, V1, V2 ... V28, Amount
    """
)

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# =========================
# PREDICTIONS
# =========================

if uploaded_file:

    data = pd.read_csv(uploaded_file)

    st.info(
        f"Transactions Uploaded: {len(data)}"
    )

    # Remove Class column if present
    if "Class" in data.columns:
        data = data.drop(
            "Class",
            axis=1
        )

    predictions = model.predict(data)

    probabilities = model.predict_proba(
        data
    )[:, 1]

    data["Prediction"] = predictions

    data["Fraud_Probability"] = probabilities

    # Risk Levels
    data["Risk_Level"] = data[
        "Fraud_Probability"
    ].apply(
        lambda x:
        "High Risk"
        if x > 0.80
        else (
            "Medium Risk"
            if x > 0.30
            else "Low Risk"
        )
    )

    st.subheader("Prediction Results")

    st.dataframe(data)

    fraud_count = (
        predictions == 1
    ).sum()

    st.success(
        f"Fraudulent Transactions Detected: {fraud_count}"
    )

    st.subheader(
        "Fraud Probability Distribution"
    )

    st.bar_chart(
        data["Fraud_Probability"]
    )

    # Download Results
    csv = data.to_csv(
        index=False
    )

    st.download_button(
        label="Download Results",
        data=csv,
        file_name="fraud_predictions.csv",
        mime="text/csv"
    )

# =========================
# FEATURE IMPORTANCE
# =========================

st.subheader(
    "Top Fraud Detection Features"
)

st.bar_chart(
    feature_importance.set_index(
        "Feature"
    )
)

st.caption(
    "Feature importance obtained from the trained XGBoost fraud detection model."
)