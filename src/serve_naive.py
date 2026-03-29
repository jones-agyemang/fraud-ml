"""
Serve fraud detection model as a REST API - NAIVE Version
"""

import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

# Load the trained model and encoder at startup
# This is loaded once when the server starts, not on every request
print("Loading model...")
with open("models/model.pkl", "rb") as f:
    model, encoder = pickle.load(f)
print("Successfully loaded model!")

# create FastAPI application
app = FastAPI(
    title="Fraud Detection API",
    description="Uses ML to detect and flag fradulent transactions"
)

class Transaction(BaseModel):
    """
    Fraud transaction schema
    """
    amount: float = Field(
        ...,
        description="Transaction amount in pounds sterling",
        example=150.00
    )
    hour: int = Field(
        ...,
        description="Hour of the day(0-23)",
        example=13
    )
    day_of_week: int = Field(
        ...,
        description="Day of week e.g. (0=Monday, ..., 6=Sunday)",
        example=3
    )
    merchant_category: str = Field(
        ...,
        description="type of merchant",
        example="travel"
    )

class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    is_fraud: bool = Field(description="returns the results for legitimacy of the transactions")
    fraud_probability: float = Field(description="Probability of fraud (0.0 to 1.0)")

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """
    Predict whether a txn is fraudulent

    Returns prediction and probability of fraud for a given request
    """
    # Convert request to a dictionary
    data = transaction.dict()
    print(f"Transaction data: {data}")

    # Encode the merchant category using the same encoder training
    # This ensures consistency between training and serving
    try:
        data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    except ValueError:
        # Handle unknown merchant categories
        data["merchant_encoded"] = 0
    
    # Prepare features in the same order as the training data
    X = [[
        data["amount"],
        data["hour"],
        data["day_of_week"],
        data["merchant_encoded"]
    ]]

    # Get prediction and probability
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] # probability of fraud (class 1)

    return PredictionResponse(
        is_fraud=bool(prediction),
        fraud_probability=round(float(probability), 4)
    )

@app.get("/health")
def health_check():
    """
    Health check endpoint.

    Returns API status. Useful for:
        - Load balancer health checks
        - Kubernetes liveness probes
        - Monitoring systems
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Fraud detetection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }