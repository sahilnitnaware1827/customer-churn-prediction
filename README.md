Sure. Copy everything below directly into your `README.md`:

````markdown
# Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn and identifying the factors that contribute most to customer attrition.

The project covers the complete Data Science workflow — from data understanding and exploratory analysis to model training, evaluation, hyperparameter tuning, explainability, and production deployment through FastAPI and Docker.

---

## 📌 Project Overview

Customer churn is a major business problem for subscription-based companies. Identifying customers who are likely to leave allows businesses to take proactive retention actions.

This project develops a binary classification model that predicts whether a customer is likely to churn based on demographic, service, contract, billing, and account-related information.

The final solution is packaged as a reusable machine learning pipeline and exposed through a REST API using FastAPI. The application can also be executed inside a Docker container.

---

## 🎯 Business Objective

The primary objectives of this project are:

- Predict whether a customer will churn.
- Identify the factors that have the greatest influence on churn.
- Compare multiple machine learning classification models.
- Optimize the selected model using hyperparameter tuning.
- Explain model predictions using SHAP.
- Build a reusable production-oriented ML pipeline.
- Expose predictions through a REST API.
- Containerize the application using Docker.

---

## 🗂️ Dataset

The project uses a customer churn dataset containing information about:

- Customer demographics
- Tenure
- Phone services
- Internet services
- Online security and backup
- Technical support
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges

### Target Variable

`Churn`

The target represents whether the customer left the service.

- `Yes` → Customer churned
- `No` → Customer did not churn

---

# 🔄 Project Workflow

The project follows an end-to-end Data Science workflow:

```text
Business Understanding
        ↓
Data Understanding
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Statistical Analysis
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Hyperparameter Tuning
        ↓
Model Explainability
        ↓
Production ML Pipeline
        ↓
Model Serialization
        ↓
Prediction Module
        ↓
FastAPI
        ↓
Input Validation
        ↓
Automated Testing
        ↓
Docker
````

---

# 🧹 Data Preparation

The data preparation stage included:

* Handling missing and inconsistent values.
* Converting data into appropriate data types.
* Separating numerical and categorical features.
* Encoding categorical variables.
* Preparing features for machine learning.
* Building reusable preprocessing logic.

The preprocessing workflow was incorporated into the production machine learning pipeline so that the same transformations are applied during prediction.

---

# 📊 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

* Customer demographics
* Churn distribution
* Contract behavior
* Tenure patterns
* Monthly charges
* Total charges
* Internet and support services
* Payment methods
* Relationships between customer characteristics and churn

EDA helped identify important patterns and potential churn drivers before model development.

---

# 🤖 Machine Learning Models

Multiple classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting

The models were evaluated using multiple classification metrics rather than relying only on accuracy.

---

# 📏 Model Evaluation

The following metrics were used:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

### Model Comparison

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression |   0.8055 |    0.6572 | 0.5588 |   0.6040 |  0.8421 |
| Decision Tree       |   0.7211 |    0.4755 | 0.4920 |   0.4836 |  0.6477 |
| Random Forest       |   0.7835 |    0.6186 | 0.4813 |   0.5414 |  0.8206 |
| Gradient Boosting   |   0.8027 |    0.6655 | 0.5160 |   0.5813 |  0.8433 |

Gradient Boosting was selected as the strongest candidate based on the overall evaluation, particularly ROC-AUC and precision.

---

# ⚙️ Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune the Gradient Boosting model.

### Search Parameters

```python
{
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}
```

### Best Parameters

```python
{
    "learning_rate": 0.05,
    "max_depth": 2,
    "n_estimators": 200
}
```

### Tuned Model Performance

| Metric    | Before Tuning | After Tuning |
| --------- | ------------: | -----------: |
| Accuracy  |      0.802697 |     0.798439 |
| Precision |      0.665517 |     0.656250 |
| Recall    |      0.516043 |     0.505348 |
| F1 Score  |      0.581325 |     0.570997 |
| ROC-AUC   |      0.843275 |     0.845508 |

The tuned model achieved a slightly higher ROC-AUC, while some threshold-dependent metrics decreased.

---

# 🔍 Model Explainability

SHAP was used to understand the contribution of individual features to the Gradient Boosting model's predictions.

### Important Features Identified

The strongest features included:

1. Contract — Month-to-month
2. Tenure
3. Internet Service — Fiber optic
4. Total Charges
5. Monthly Charges
6. Online Security — No
7. Payment Method — Electronic check
8. Tech Support — No
9. Paperless Billing
10. Multiple Lines

This provides insight into which customer characteristics are most influential in the model's churn predictions.

---

# 🏗️ Production ML Pipeline

The trained machine learning workflow was converted into reusable production-oriented components.

The pipeline includes:

```text
Raw Customer Data
       ↓
Preprocessing
       ↓
Feature Transformation
       ↓
Gradient Boosting Model
       ↓
Prediction
```

The complete trained pipeline is serialized using Joblib.

This allows the same preprocessing and model logic to be reused during inference.

---

# 🔮 Prediction Module

A reusable prediction module was created in:

```text
src/predict.py
```

The prediction module:

* Loads the serialized ML pipeline.
* Accepts customer information.
* Applies the required preprocessing.
* Generates the churn prediction.
* Calculates churn probability.
* Returns the prediction result.

Example response:

```json
{
    "churn_prediction": "Yes",
    "churn_probability": 0.78
}
```

The exact probability depends on the input customer and trained model.

---

# 🚀 FastAPI

The trained model is exposed through a REST API using FastAPI.

### API Endpoints

#### Health Check

```text
GET /
```

Response:

```json
{
    "message": "Customer Churn Prediction API is running"
}
```

#### Churn Prediction

```text
POST /predict
```

The endpoint accepts customer information and returns the predicted churn class and probability.

---

# 🛡️ Input Validation

Pydantic is used to validate API input before it reaches the machine learning model.

Validation includes:

* Allowed categorical values
* Numeric data types
* Tenure range
* Monthly charge validation
* Total charge validation
* Required fields

Invalid requests are rejected by the API with an appropriate validation response.

---

# 🧪 Testing

Automated API tests were implemented using Pytest.

Tests cover:

* API health check
* Valid prediction request
* Invalid input validation

Run the tests using:

```bash
pytest
```

Expected result:

```text
3 passed
```

---

# 🐳 Docker

The FastAPI application is containerized using Docker.

### Build Docker Image

```bash
docker build -t customer-churn-api .
```

### Run Container

```bash
docker run -p 8000:8000 customer-churn-api
```

The API can then be accessed through:

```text
http://localhost:8000
```

Swagger API documentation:

```text
http://localhost:8000/docs
```

---

# 🌐 Live Deployment

The Customer Churn Prediction API is deployed on Render using Docker.

### Live API

https://customer-churn-prediction-zpj7.onrender.com

### Swagger API Documentation

https://customer-churn-prediction-zpj7.onrender.com/docs

The live API allows users to submit customer information and receive a churn prediction and probability through the FastAPI REST API.

---

# 📁 Project Structure

```text
customer-churn-prediction/
│
├── api/
│   └── main.py
│
├── src/
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── churn_pipeline.joblib
│
├── tests/
│   └── test_api.py
│
├── notebooks/
│   ├── ...
│
├── data/
│   └── ...
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* SHAP

### Machine Learning

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* GridSearchCV
* Cross-validation

### Backend

* FastAPI
* Pydantic
* Uvicorn

### Deployment & Engineering

* Docker
* Joblib
* Pytest
* Git
* GitHub

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sahilnitnaware1827/customer-churn-prediction
```

Move into the project:

```bash
cd customer-churn-prediction
```

Create a virtual environment:

```bash
python -m venv DSvenv
```

Activate it on Windows:

```powershell
.\DSvenv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the API Locally

From the project root:

```bash
uvicorn api.main:app --reload
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 🐳 Run with Docker

Build:

```bash
docker build -t customer-churn-api .
```

Run:

```bash
docker run -p 8000:8000 customer-churn-api
```

Open:

```text
http://localhost:8000/docs
```

---

# 🧪 Run Tests

```bash
pytest
```

---

# 💡 Key Learning Outcomes

This project demonstrates practical experience across the complete machine learning lifecycle:

* Translating a business problem into an ML problem
* Data cleaning and exploratory analysis
* Statistical analysis
* Feature engineering
* Supervised machine learning
* Model comparison
* Model evaluation
* Hyperparameter optimization
* Model explainability
* Reusable preprocessing pipelines
* Model serialization
* REST API development
* Input validation
* Automated testing
* Docker containerization
* Git/GitHub project management

---

# 🔮 Future Improvements

Potential improvements include:

* Model monitoring
* Data drift detection
* Experiment tracking
* CI/CD pipeline
* Authentication and authorization
* Logging and monitoring
* Automated model retraining
* Advanced model calibration
* Performance monitoring in production

---

# 👨‍💻 Author

**Sahil Nitnaware**

Data Scientist | Machine Learning | Data Analytics

GitHub:
[https://github.com/sahilnitnaware1827](https://github.com/sahilnitnaware1827)

LinkedIn:
[www.linkedin.com/in/sahil-nitnaware-0540b1252](http://www.linkedin.com/in/sahil-nitnaware-0540b1252)
