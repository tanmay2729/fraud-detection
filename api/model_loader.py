import joblib
import numpy as np

# Simple hardcoded path — works reliably on Windows
model = joblib.load(r"D:\fraud-detection\notebooks\models\fraud_model.pkl")
scaler = joblib.load(r"D:\fraud-detection\notebooks\models\scaler.pkl")

def predict_transaction(features: list) -> dict:
    features_array = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0][1]
    
    return {
        "prediction": int(prediction),
        "fraud_probability": round(float(probability), 4),
        "verdict": "FRAUD" if prediction == 1 else "LEGITIMATE"
    }