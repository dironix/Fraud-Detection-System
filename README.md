# Credit Card Fraud Detection System

## Overview

This project uses Machine Learning to detect fraudulent credit card transactions. Due to the highly imbalanced nature of fraud datasets, SMOTE oversampling was applied to improve fraud detection performance.

The system includes model explainability, performance evaluation, and a Streamlit web application for real-time prediction.

---

## Dataset

Credit Card Fraud Detection Dataset

Source:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset Size:

* 284,807 transactions
* 492 fraudulent transactions
* 30 anonymized PCA-transformed features

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* SHAP
* Streamlit
* Matplotlib
* Seaborn

---

## Machine Learning Pipeline

1. Data Loading
2. Exploratory Data Analysis
3. Train-Test Split
4. SMOTE Oversampling
5. Random Forest Training
6. XGBoost Training
7. Model Evaluation
8. Feature Importance Analysis
9. Model Deployment with Streamlit

---

## Best Model: XGBoost

### Performance Metrics

| Metric    | Score  |
| --------- | ------ |
| Precision | 0.79   |
| Recall    | 0.85   |
| F1 Score  | 0.82   |
| ROC-AUC   | 0.9831 |

---

## Top Important Features

| Feature | Importance |
| ------- | ---------- |
| V14     | 0.7027     |
| V4      | 0.0562     |
| V12     | 0.0302     |
| V17     | 0.0206     |
| V3      | 0.0155     |
| V1      | 0.0133     |
| V8      | 0.0133     |
| V13     | 0.0121     |
| V10     | 0.0094     |
| V7      | 0.0092     |

---

## Streamlit Application

Features:

* Upload transaction CSV
* Fraud prediction
* Fraud probability scoring
* Download prediction results
* Feature importance visualization
* Performance metrics dashboard

---

## Project Structure

Fraud-Detection-System/

├── app/

├── data/

├── models/

├── notebooks/

├── src/

├── README.md

└── requirements.txt

---

## Run Locally

pip install -r requirements.txt

streamlit run app/app.py

---

## Author

Diya Sarkar
