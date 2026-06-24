from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from api.model_loader import predict_transaction

# Initialize app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection using Random Forest",
    version="1.0.0"
)

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define what a transaction looks like
class Transaction(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 30  # 30 feature values
            }
        }

# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running!",
        "status": "healthy"
    }

# Prediction endpoint
@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        return {
            "error": f"Expected 30 features, got {len(transaction.features)}"
        }
    
    result = predict_transaction(transaction.features)
    return result