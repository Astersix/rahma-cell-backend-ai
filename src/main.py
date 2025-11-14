from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

# konfigurasi
N_STEPS = 30
PRODUCT_ID = 'A001'
MODEL_PATH = "models"

class PredictRequest(BaseModel):
    history: List[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("server is loading the model...") 
    
    try:
        app.state.models = {}
        app.state.models['svr_model'] = joblib.load(
            os.path.join(MODEL_PATH, f"svr_model_{PRODUCT_ID}.joblib")
        )
        app.state.models['scaler_X'] = joblib.load(
            os.path.join(MODEL_PATH, f"scaler_X_{PRODUCT_ID}.joblib")
        )
        app.state.models['scaler_y'] = joblib.load(
            os.path.join(MODEL_PATH, f"scaler_y_{PRODUCT_ID}.joblib")
        )
        print("successfully load model and scaler on to app.state")
        
    except Exception as e:
        print(f"failed to load model when startup: {e}")
        app.state.models = {}
    
    yield
    
    print("turning off server...")
    if hasattr(app.state, 'models'):
        app.state.models.clear()
        
    print("finished cleanup")
    
app = FastAPI(lifespan=lifespan)

@app.get("/")
def readRoot():
    return {
        "message": f"Stock prediction for product: {PRODUCT_ID}"
    }

@app.post("/predict/")
async def predictStock(request_data: PredictRequest, req: Request):
    history_data = request_data.history
    
    if len(history_data) != N_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"History data should be filled with {N_STEPS} data points."
        )
        
    try:
        model = req.app.state.models['svr_model']
        scaler_X = req.app.state.models['scaler_X']
        scaler_y = req.app.state.models['scaler_y']
    except (AttributeError, KeyError):
        raise HTTPException(
            status_code=503,
            detail="Server is initiating or model failed to load."
        )
    
    try:
        input_array = np.array(history_data).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Data tidak valid. Error: {e}")

    input_scaled = scaler_X.transform(input_array)
    
    predicted_scaled = model.predict(input_scaled)

    predicted_real = scaler_y.inverse_transform(predicted_scaled.reshape(-1, 1))

    prediction_value = predicted_real[0][0]
    prediction_rounded = round(prediction_value)

    return {
        "product_id": PRODUCT_ID,
        "input_history": history_data,
        "prediction_raw": prediction_value,
        "prediction_rounded": prediction_rounded
    }
