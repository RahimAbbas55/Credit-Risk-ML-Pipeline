from fastapi import FastAPI, HTTPException
from api.schemas import LoanApplicationRequest, PredictionResponse
from api.prediction import predict_default_risk

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts loan default risk using a tuned XGBoost model trained on the Home Credit Default Risk dataset.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Credit Risk Prediction API is running. See /docs for usage."}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: LoanApplicationRequest):
    try:
        result = predict_default_risk(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")