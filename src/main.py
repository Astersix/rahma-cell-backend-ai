from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np
from datetime import datetime, timedelta
import os
from functools import lru_cache
import logging
import psutil
import pandas as pd

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = "models/"

# schema request
class StockPredictionRequest(BaseModel):
    product_variant_id: str 
    price: float            
    category_id: int        
    last_transaction_date: str 
    sales_history_30_days: List[float] 

# helper untuk cek memori
def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024 # convert to mb
    logger.info(f"server memory usage: {mem:.2f} mb")

# model loader dengan lru cache
@lru_cache(maxsize=100)
def load_model_resources(variant_id: str):
    logger.info(f"cache miss: loading resources from disk for variant {variant_id}")
    
    path_model = os.path.join(MODEL_PATH, f'svr_model_{variant_id}.joblib')
    path_scaler_x = os.path.join(MODEL_PATH, f'scaler_X_{variant_id}.joblib')
    path_scaler_y = os.path.join(MODEL_PATH, f'scaler_y_{variant_id}.joblib')

    if not (os.path.exists(path_model) and os.path.exists(path_scaler_x) and os.path.exists(path_scaler_y)):
        logger.error(f"files not found for variant {variant_id}")
        return None

    try:
        model = joblib.load(path_model)
        scaler_X = joblib.load(path_scaler_x)
        scaler_y = joblib.load(path_scaler_y)
        return model, scaler_X, scaler_y
    except Exception as e:
        logger.error(f"error loading joblib files: {e}")
        return None

@app.post("/predict-stock")
def predict_stock(request: StockPredictionRequest):
    # catat memori saat request masuk
    log_memory_usage()
    
    variant_id = request.product_variant_id
    logger.info(f"received prediction request for {variant_id}")

    # validasi input data history
    history = request.sales_history_30_days
    
    if not history:
        logger.error("history is empty or none")
        raise HTTPException(status_code=400, detail="sales history cannot be empty")
        
    if len(history) < 30:
        logger.warning(f"insufficient history length: {len(history)}")
        raise HTTPException(status_code=400, detail=f"sales history must be at least 30 days. received {len(history)}")
    
    if any(x < 0 for x in history):
        logger.error("negative value detected in sales history")
        raise HTTPException(status_code=400, detail="sales history cannot contain negative numbers")

    # load resources
    resources = load_model_resources(variant_id)
    
    if resources is None:
        raise HTTPException(status_code=404, detail=f"model for variant {variant_id} not found")
    
    model, scaler_X, scaler_y = resources
    
    # validasi format tanggal
    try:
        current_date = datetime.strptime(request.last_transaction_date, "%Y-%m-%d")
    except ValueError:
        logger.error(f"invalid date format received: {request.last_transaction_date}")
        raise HTTPException(status_code=400, detail="invalid date format. use yyyy-mm-dd")
    
    predictions = []
    temp_history = list(history) # copy agar tidak mengubah data asli request
    
    try:
        # iterative prediction 7 days
        for i in range(7):
            next_date = current_date + timedelta(days=i+1)
            
            # feature engineering real-time
            day_of_week = next_date.weekday()
            day_of_month = next_date.day
            is_weekend = 1 if day_of_week >= 5 else 0
            
            # rolling calculation
            lag_1 = temp_history[-1]
            lag_7 = temp_history[-7]
            
            current_window_7 = temp_history[-7:]
            rolling_mean_7 = np.mean(current_window_7)
            # handle deviasi standar jika data konstan
            rolling_std_7 = np.std(current_window_7, ddof=1) if len(current_window_7) > 1 else 0.0
            
            # Pastikan urutan nama kolom SAMA PERSIS dengan di train.py
            feature_names = ['price', 'category_id', 'day_of_week', 'day_of_month', 'is_weekend', 
                             'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
            
            # Buat DataFrame, bukan NumPy array
            features_df = pd.DataFrame([[
                request.price,
                request.category_id,
                day_of_week,
                day_of_month,
                is_weekend,
                lag_1,
                lag_7,
                rolling_mean_7,
                rolling_std_7
            ]], columns=feature_names)
            
            # predict
            features_scaled = scaler_X.transform(features_df)
            pred_scaled = model.predict(features_scaled)
            pred_value = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
            
            pred_value = max(0, pred_value)
            
            predictions.append(pred_value)
            temp_history.append(pred_value)
            
    except Exception as e:
        logger.critical(f"prediction logic failed: {e}")
        raise HTTPException(status_code=500, detail="internal calculation error")
    
    total_restock = sum(predictions)
    
    logger.info(f"prediction success for {variant_id}. result: {total_restock}")
    
    return {
        "product_variant_id": variant_id,
        "prediction_period": "next 7 days",
        "daily_predictions": [round(x, 2) for x in predictions],
        "total_restock_recommended": int(np.ceil(total_restock))
    }