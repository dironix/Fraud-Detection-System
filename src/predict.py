import joblib
import pandas as pd

model = joblib.load(
    "../models/random_forest_model.pkl"
)

def predict_transaction(transaction):

    df = pd.DataFrame([transaction])

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability