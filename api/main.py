import os
import sys
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.Fraud_Detection.exception.exceptions import CustomException
from src.Fraud_Detection.logger.logger import logger
from src.Fraud_Detection.utils.common import load_object

app = FastAPI(title="Financial Fraud Detection")

@app.get("/")
def index():
    return {"message": "Welcome to the Financial Fraud Detection API!"}

@app.exception_handler(CustomException)
async def custom_exception_handler(request: Request, exc: CustomException):
    logger.error(f"CustomException: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": exc.error_message},
    )

model = load_object("artifacts/model.pkl")
preprocessor = load_object("artifacts/preprocessor.pkl")

# custom data model for input validation
class CustomData(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict(data: CustomData):
    try:
        input_data = np.array([[
            data.Time, data.V1, data.V2, data.V3, data.V4, data.V5, data.V6, data.V7, data.V8, data.V9,
            data.V10, data.V11, data.V12, data.V13, data.V14, data.V15, data.V16, data.V17, data.V18,
            data.V19, data.V20, data.V21, data.V22, data.V23, data.V24, data.V25, data.V26, data.V27,
            data.V28, data.Amount
        ]])
        columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
                   'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
                   'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27',
                   'V28', 'Amount']


        BEST_THRESHOLD = 0.3

        input_df = pd.DataFrame(input_data, columns=columns)

        # 1. Transform input
        data_scaled = preprocessor.transform(input_df)

        # 2. Get probability first
        probability = model.predict_proba(data_scaled)[:, 1]

        # 3. Apply threshold to get prediction
        prediction = int(probability[0] >= BEST_THRESHOLD)

        # 4. Determine risk
        risk = "High" if probability[0] >= BEST_THRESHOLD else "Low"

        # 5. Return
        return {
            "prediction": prediction,
            "probability": float(probability[0]),
            "risk": risk
        }
    except Exception as e:
        raise CustomException(e, sys)