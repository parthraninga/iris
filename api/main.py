from fastapi import FastAPI, HTTPException
from joblib import load
from api.schemas import IrisRequest, PredictionResponse
import numpy as np

app = FastAPI(title="Iris ML Model API")

# Load model at startup
try:
    model = load("iris_pipeline_model.pkl")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

@app.get("/")
def root():
    return {"message": "Welcome to the Iris Classifier API 🚀"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: IrisRequest):
    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()
        return PredictionResponse(prediction=prediction, probabilities=probabilities)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
