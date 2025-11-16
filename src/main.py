from fastapi import FastAPI, HTTPException, Request
from functools import lru_cache
# from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import os

# konfigurasi
N_STEPS = 90
MODEL_PATH = "models"
PREDICTION_HORIZON = 30

@lru_cache(maxsize=100)
def getModelResource(product_variant_id: str):
    path_model = os.path.join(MODEL_PATH, f"svr_model_{product_variant_id}.joblib")
    path_scaler_x = os.path.join(MODEL_PATH, f"scaler_X_{product_variant_id}.joblib")
    path_scaler_y = os.path.join(MODEL_PATH, f"scaler_y_{product_variant_id}.joblib")
    
    if not os.path.exists(path_model):
        return None

    try:
        model = joblib.load(path_model)
        scaler_x = joblib.load(path_scaler_x)
        scaler_y = joblib.load(path_scaler_y)
        return model, scaler_x, scaler_y
    
    except Exception as e:
        print(f"Error loading model {product_variant_id}: {e}")
        raise RuntimeError(f"Gagal memuat model {product_variant_id}")

class PredictRequest(BaseModel):
    product_variant_id: str
    history: List[float]

app = FastAPI()

@app.post("/predict/")
async def predictStock(request_data: PredictRequest, req: Request):
    product_variant_id = request_data.product_variant_id
    history = request_data.history
    
    if len(history) < N_STEPS:
        raise HTTPException(
            status_code=400,
            detail=f"History data should be filled with at least {N_STEPS} data points to make a prediction."
        )
        
    resources = getModelResource(product_variant_id)
    
    if resources is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Model for product variant: '{product_variant_id}' not found."
        )
    
    model, scaler_X, scaler_y = resources
    
    current_window = list(history[-N_STEPS:])
    
    predictions_list = []
    
    for _ in range(PREDICTION_HORIZON):
        window_np = np.array(current_window).reshape(1, -1)
        
        window_scaled = scaler_X.transform(window_np)
        
        pred_scaled = model.predict(window_scaled)
        pred_real_np = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        
        pred_real_float = pred_real_np[0][0]
        
        pred_cleaned = round(pred_real_float)
        pred_cleaned = max(0, pred_cleaned)
        
        predictions_list.append(pred_cleaned)
        
        current_window.pop(0)
        current_window.append(pred_cleaned)
        
    return {
        "product_variant_id": product_variant_id,
        "prediction": predictions_list
    }
