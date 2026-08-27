import joblib
import pandas as pd


MODEL_PATH = "models/churn_pipeline.joblib"


def predict_churn(customer_data: dict):

    pipeline = joblib.load(MODEL_PATH)

    customer_df = pd.DataFrame([customer_data])

    prediction = pipeline.predict(customer_df)[0]

    probabilities = pipeline.predict_proba(customer_df)[0]

    yes_index = list(pipeline.classes_).index("Yes")
    probability = probabilities[yes_index]

    return {
        "churn_prediction": prediction,
        "churn_probability": float(probability)
    }


if __name__ == "__main__":

    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.5,
        "TotalCharges": 427.5
    }

    result = predict_churn(customer)

    print(result)
