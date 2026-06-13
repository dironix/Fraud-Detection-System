# Fraud Detection System using Machine Learning

## Project Overview

This project aims to detect fraudulent credit card transactions using Machine Learning techniques. Due to the highly imbalanced nature of fraud datasets, the project employs SMOTE (Synthetic Minority Oversampling Technique) to improve the model's ability to identify fraudulent transactions.

The system includes:

* Data preprocessing and exploration
* Class imbalance handling using SMOTE
* Random Forest model training
* Model evaluation using classification metrics and ROC-AUC
* Model explainability using SHAP
* Streamlit web application for fraud prediction

---

## Problem Statement

Detect fraudulent credit card transactions accurately while minimizing false negatives and false positives.

---

## Dataset

**Credit Card Fraud Detection Dataset**

* Source: Kaggle
* Total Transactions: 284,807
* Fraudulent Transactions: 492
* Legitimate Transactions: 284,315

The dataset contains anonymized PCA-transformed features (V1–V28), along with:

* Time
* Amount
* Class (Target Variable)

Target Variable:

* 0 → Legitimate Transaction
* 1 → Fraudulent Transaction

---

## Project Structure

Fraud-Detection-System/

├── data/

│ ├── raw/

│ └── processed/

├── models/

│ └── random_forest_model.pkl

├── notebooks/

├── src/

│ └── predict.py

├── app/

│ └── app.py

├── requirements.txt

├── README.md

└── .gitignore

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* SHAP
* Matplotlib
* Streamlit
* Joblib

---

## Machine Learning Pipeline

### Data Preprocessing

* Loaded and explored transaction data
* Checked for missing values
* Analyzed class imbalance

### Train-Test Split

* 80% Training Data
* 20% Testing Data

### Class Imbalance Handling

SMOTE was applied to the training dataset to balance fraudulent and non-fraudulent transactions.

### Model Training

Random Forest Classifier

Parameters:

* n_estimators = 200
* random_state = 42
* n_jobs = -1

---

## Model Performance

| Metric    | Value  |
| --------- | ------ |
| Precision | 0.79   |
| Recall    | 0.85   |
| F1-Score  | 0.82   |
| ROC-AUC   | 0.9831 |

Confusion Matrix:

|                   | Predicted Legitimate | Predicted Fraud |
| ----------------- | -------------------- | --------------- |
| Actual Legitimate | 56809                | 55              |
| Actual Fraud      | 13                   | 85              |

---

## Model Explainability (SHAP)

Top Features Influencing Fraud Predictions:

| Feature | Importance |
| ------- | ---------- |
| V14     | 0.0791     |
| V12     | 0.0739     |
| V4      | 0.0675     |
| V3      | 0.0456     |
| V10     | 0.0446     |
| V11     | 0.0366     |
| V17     | 0.0313     |
| V16     | 0.0189     |
| V1      | 0.0114     |
| V7      | 0.0101     |

---

## Streamlit Application

Run the application:

```bash
streamlit run app/app.py
```

The application allows users to upload transaction datasets and identify potentially fraudulent transactions.

---

## Future Improvements

* XGBoost and LightGBM comparison
* Real-time fraud detection API
* Advanced Streamlit dashboard
* Transaction risk scoring
* Explainable AI visualizations

---

## Author

Diya Sarkar
