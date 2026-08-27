import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor


DATA_PATH = "data/cleaned/telco_customer_churn_cleaned.csv"
MODEL_PATH = "models/churn_pipeline.joblib"


def train_model():

    # Load cleaned data
    df = pd.read_csv(DATA_PATH)

    # Remove ID column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Separate features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Create preprocessing
    preprocessor = create_preprocessor(X)

    # Final model
    model = GradientBoostingClassifier(
        random_state=42,
        learning_rate=0.05,
        max_depth=2,
        n_estimators=200
    )

    # Complete production pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X, y)

    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
    