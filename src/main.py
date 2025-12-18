from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
from functools import lru_cache
import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from filelock import FileLock

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# setup rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

MODEL_PATH = 'models/'
LOCK_FILE = "models/.retrain.lock"

class StockPredictionRequest(BaseModel):
    product_variant_id: str 
    price: float            
    category_id: int        
    last_transaction_date: str 
    sales_history_30_days: List[float] 

@lru_cache(maxsize=100)
def load_model_resources(variant_id: str):
    path_model = os.path.join(MODEL_PATH, f'svr_model_{variant_id}.joblib')
    path_scaler_x = os.path.join(MODEL_PATH, f'scaler_X_{variant_id}.joblib')
    
    if not (os.path.exists(path_model) and os.path.exists(path_scaler_x)):
        return None

    try:
        model = joblib.load(path_model)
        scaler_X = joblib.load(path_scaler_x)
        return model, scaler_X
    except Exception as e:
        logger.error(f"error loading files: {e}")
        return None

@app.post("/predict-stock")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute per IP
def predict_stock(request: StockPredictionRequest):
    variant_id = request.product_variant_id
    
    # Check if retraining is in progress
    if os.path.exists(LOCK_FILE):
        try:
            lock = FileLock(LOCK_FILE, timeout=0.1)
            lock.acquire()
            lock.release()
        except:
            raise HTTPException(
                status_code=503,
                detail="System is currently retraining models. Please try again in a few moments."
            )
    
    # load resources
    resources = load_model_resources(variant_id)
    if resources is None:
        raise HTTPException(status_code=404, detail=f"model for {variant_id} not found")
    
    model, scaler_X = resources
    
    history = list(request.sales_history_30_days)
    temp_history = list(history)
    
    try:
        current_date = datetime.strptime(request.last_transaction_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid date format")
    
    predictions = []
    
    for i in range(7):
        next_date = current_date + timedelta(days=i+1)
        
        # feature engineering real-time
        day_of_week = next_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # rolling calculation
        lag_1 = temp_history[-1]
        lag_7 = temp_history[-7]
        
        current_window_7 = temp_history[-7:]
        rolling_mean_7 = np.mean(current_window_7)
        rolling_std_7 = np.std(current_window_7, ddof=1) if len(current_window_7) > 1 else 0.0
        
        input_features = pd.DataFrame([{
            'price': request.price,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'lag_1': lag_1,
            'lag_7': lag_7,
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_7': rolling_std_7
        }])
        
        # scaling & predict
        features_scaled = scaler_X.transform(input_features)
        
        # hasil prediksi ini dalam bentuk LOG
        pred_log = model.predict(features_scaled)[0]
        
        # kembalikan ke bentuk asli (inverse log)
        pred_value = np.expm1(pred_log)
        
        pred_value = max(0, pred_value)
        
        predictions.append(pred_value)
        temp_history.append(pred_value)
            
    total_restock = sum(predictions)
    
    # Log predictions for debugging
    logger.info(f"Raw predictions for {variant_id}: {predictions}")
    logger.info(f"Rounded predictions: {[round(x, 2) for x in predictions]}")
    
    return {
        "product_variant_id": variant_id,
        "prediction_period": "next 7 days",
        "daily_predictions": [round(x, 2) for x in predictions],
        "total_restock_recommended": int(np.ceil(total_restock))
    }