from fastapi import FastAPI, HTTPException, Request
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
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# setup rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter

MODEL_PATH = 'models/'
LOCK_FILE = "models/.retrain.lock"

# Setup database connection
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base = automap_base()
    Base.prepare(autoload_with=engine)
    
    # Access to tables
    product_variant = Base.classes.product_variant
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Database connection failed - {str(e)}")
    DB_AVAILABLE = False

class StockPredictionRequest(BaseModel):
    product_variant_id: str 
    price: float            
    category_id: int        
    last_transaction_date: str 
    sales_history_30_days: List[float]
    current_stock: float = 0

def get_current_stock(variant_id: str) -> float:
    if not DB_AVAILABLE:
        logger.warning(f"Database not available, using default stock")
        return 0
    
    try:
        with Session(engine) as session:
            result = session.query(product_variant.stock).filter(
                product_variant.id == variant_id
            ).first()
            
            if result:
                return float(result[0]) if result[0] is not None else 0
            else:
                logger.warning(f"Product variant {variant_id} not found in database")
                return 0
    except Exception as e:
        logger.error(f"Error fetching stock for {variant_id}: {str(e)}")
        return 0

@lru_cache(maxsize=100)
def load_model_resources(variant_id: str):
    path_model = os.path.join(MODEL_PATH, f'svr_model_{variant_id}.joblib')
    path_scaler_x = os.path.join(MODEL_PATH, f'scaler_X_{variant_id}.joblib')
    path_features = os.path.join(MODEL_PATH, f'features_{variant_id}.joblib')
    
    if not (os.path.exists(path_model) and os.path.exists(path_scaler_x) and os.path.exists(path_features)):
        return None

    try:
        model = joblib.load(path_model)
        scaler_X = joblib.load(path_scaler_x)
        features = joblib.load(path_features)
        return model, scaler_X, features
    except Exception as e:
        logger.error(f"error loading files: {e}")
        return None

@app.post("/predict-stock")
@limiter.limit("30/minute")  # Rate limit: 30 requests per minute per IP
def predict_stock(request: Request, stock_request: StockPredictionRequest):
    variant_id = stock_request.product_variant_id
    
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
    
    # load resources with features list
    resources = load_model_resources(variant_id)
    if resources is None:
        raise HTTPException(status_code=404, detail=f"model for {variant_id} not found")
    
    model, scaler_X, features = resources
    
    # Get current stock from database if not provided in request
    current_stock = stock_request.current_stock
    if current_stock == 0:
        current_stock = get_current_stock(variant_id)
    
    history = list(stock_request.sales_history_30_days)
    temp_history = list(history)
    
    try:
        current_date = datetime.strptime(stock_request.last_transaction_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid date format")
    
    predictions = []
    
    for i in range(7):
        next_date = current_date + timedelta(days=i+1)
        
        # Create all possible features
        day_of_week = next_date.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0
        day = next_date.day
        
        # lag features - create all possible lags
        lag_1 = temp_history[-1] if len(temp_history) >= 1 else 0
        lag_2 = temp_history[-2] if len(temp_history) >= 2 else 0
        lag_3 = temp_history[-3] if len(temp_history) >= 3 else 0
        lag_4 = temp_history[-4] if len(temp_history) >= 4 else 0
        lag_5 = temp_history[-5] if len(temp_history) >= 5 else 0
        lag_6 = temp_history[-6] if len(temp_history) >= 6 else 0
        lag_7 = temp_history[-7] if len(temp_history) >= 7 else 0
        
        # rolling statistics
        current_window_3 = temp_history[-3:] if len(temp_history) >= 3 else temp_history
        current_window_7 = temp_history[-7:] if len(temp_history) >= 7 else temp_history
        
        rolling_mean_3 = np.mean(current_window_3) if current_window_3 else 0
        rolling_mean_7 = np.mean(current_window_7) if current_window_7 else 0
        rolling_std_3 = np.std(current_window_3, ddof=1) if len(current_window_3) > 1 else 0.0
        rolling_std_7 = np.std(current_window_7, ddof=1) if len(current_window_7) > 1 else 0.0
        rolling_min_3 = np.min(current_window_3) if current_window_3 else 0
        rolling_max_3 = np.max(current_window_3) if current_window_3 else 0
        
        # momentum and velocity
        momentum = temp_history[-1] - temp_history[-2] if len(temp_history) >= 2 else 0
        momentum_3 = temp_history[-1] - temp_history[-4] if len(temp_history) >= 4 else 0
        
        # trend (number of days in history)
        trend = len(temp_history)
        
        # Create dataframe with all possible features
        all_features_df = pd.DataFrame([{
            'price': stock_request.price,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'day': day,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_3': lag_3,
            'lag_4': lag_4,
            'lag_5': lag_5,
            'lag_6': lag_6,
            'lag_7': lag_7,
            'rolling_mean_3': rolling_mean_3,
            'rolling_mean_7': rolling_mean_7,
            'rolling_std_3': rolling_std_3,
            'rolling_std_7': rolling_std_7,
            'rolling_min_3': rolling_min_3,
            'rolling_max_3': rolling_max_3,
            'momentum': momentum,
            'momentum_3': momentum_3,
            'trend': trend
        }])
        
        # Select only the features that were used during training
        # This handles if training used fewer features
        input_features = all_features_df[[f for f in features if f in all_features_df.columns]]
        
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
    
    # Subtract current stock from total recommendation
    # total_restock_recommended is the additional stock needed beyond current stock
    total_restock_recommended = max(0, total_restock - current_stock)
    
    # Log predictions for debugging
    logger.info(f"Raw predictions for {variant_id}: {predictions}")
    logger.info(f"Rounded predictions: {[round(x, 2) for x in predictions]}")
    logger.info(f"Total 7-day forecast: {total_restock:.2f}, Current stock: {current_stock}, Total restock recommended: {total_restock_recommended:.2f}")
    
    return {
        "product_variant_id": variant_id,
        "prediction_period": "next 7 days",
        "daily_predictions": [round(x, 2) for x in predictions],
        "total_restock_recommended": int(np.ceil(total_restock_recommended))
    }