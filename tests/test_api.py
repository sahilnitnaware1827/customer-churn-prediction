from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


valid_customer = {
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


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Customer Churn Prediction API is running"


def test_valid_prediction():
    response = client.post(
        "/predict",
        json=valid_customer
    )

    assert response.status_code == 200

    result = response.json()

    assert "churn_prediction" in result
    assert "churn_probability" in result


def test_invalid_tenure():
    invalid_customer = valid_customer.copy()
    invalid_customer["tenure"] = 100

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422
    