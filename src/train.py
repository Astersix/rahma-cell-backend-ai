import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# configuration
MODEL_PATH = 'models/'
MIN_DATA_POINTS = 30

# Setup database connection
try:
    engine = create_engine(os.getenv('DATABASE_URL'))
    Base = automap_base()
    Base.prepare(autoload_with=engine)
    
    # Access to tables
    order = Base.classes.order
    order_product = Base.classes.order_product
    DB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Database connection failed - {str(e)}")
    DB_AVAILABLE = False 

def fetch_training_data(product_variant_id=None):
    if not DB_AVAILABLE:
        raise Exception("Database connection not available")
    
    with Session(engine) as session:
        query = session.query(
            order_product.product_variant_id,
            order.created_at.label('date'),
            order_product.quantity,
            order_product.price,
        ).join(order, order.id == order_product.order_id).filter(order.status == 'selesai')

        if product_variant_id:
            query = query.filter(order_product.product_variant_id == product_variant_id)
        
        query = query.order_by(order.created_at.asc())

        # Convert to DataFrame
        df = pd.read_sql(query.statement, session.bind)
        
        # Convert all column names to strings
        df.columns = df.columns.astype(str)
        
        # Ensure date is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Extract only date (remove time) for daily aggregation
        df['date'] = df['date'].dt.date
        
        # Aggregate by date: sum quantities, average prices
        df_daily = df.groupby(['product_variant_id', 'date']).agg({
            'quantity': 'sum',
            'price': 'mean'
        }).reset_index()
        
        # Convert date back to datetime for feature engineering
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        
        print(f"Fetched {len(df)} total records, aggregated to {len(df_daily)} daily records")
        return df_daily

def feature_engineering(df):
    df = df.sort_values('date').reset_index(drop=True)
    
    # time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day'] = df['date'].dt.day
    
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['quantity'].shift(lag).bfill().fillna(0)
    
    # rolling statistics - smaller windows for small datasets
    df['rolling_mean_3'] = df['quantity'].shift(1).rolling(window=3, min_periods=1).mean()
    df['rolling_mean_7'] = df['quantity'].shift(1).rolling(window=7, min_periods=1).mean()
    df['rolling_std_3'] = df['quantity'].shift(1).rolling(window=3, min_periods=1).std()
    df['rolling_min_3'] = df['quantity'].shift(1).rolling(window=3, min_periods=1).min()
    df['rolling_max_3'] = df['quantity'].shift(1).rolling(window=3, min_periods=1).max()
    
    # momentum and velocity
    df['momentum'] = df['quantity'].diff().fillna(0)
    df['momentum_3'] = df['quantity'].diff(3).fillna(0)
    
    # trend
    df['trend'] = np.arange(len(df))
    
    # Final fill for all NaN values
    df = df.fillna(0)
    
    # Ensure all column names are strings
    df.columns = df.columns.astype(str)
    
    return df

def train_variant(df_variant, variant_id):
    print(f"\nprocessing variant: {variant_id}")
    
    if len(df_variant) < MIN_DATA_POINTS:
        print(f"skip: insufficient data ({len(df_variant)} rows)")
        return None

    df_processed = feature_engineering(df_variant.copy())
    
    if len(df_processed) < 20:
        print("skip: insufficient data after processing")
        return None

    # Select features dynamically based on available columns
    available_features = [col for col in df_processed.columns 
                         if col not in ['date', 'product_variant_id', 'quantity']]
    
    # Use all available features
    features = available_features
    target = 'quantity'
    
    X = df_processed[features]
    y = df_processed[target]
    
    print(f"Training with {len(X)} samples, {len(features)} features")
    print(f"Features: {features}")
    
    # Use log transformation for target
    y_log = np.log1p(y)
    
    # Increase test size for better evaluation with small datasets
    test_size = 0.35 if len(df_processed) > 25 else 0.2
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=test_size, shuffle=False
    )
    
    # Scaling
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Optimized hyperparameters untuk dataset kecil
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.01, 0.1, 1],
        'epsilon': [0.01, 0.05, 0.1, 0.5]
    }
    
    svr = SVR(kernel='rbf')
    # Reduce CV folds for small datasets
    cv_folds = min(3, len(X_train) // 3)
    grid_search = GridSearchCV(
        svr, param_grid, cv=cv_folds, 
        scoring='r2', n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train_log)
    
    best_model = grid_search.best_estimator_
    print(f"best params: {grid_search.best_params_}")
    print(f"best cv score: {grid_search.best_score_:.4f}")
    
    # Evaluation
    y_pred_log = best_model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test_log)
    
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, y_pred)
    
    print(f"success. mse: {mse:.4f}, rmse: {rmse:.4f}, r2: {r2:.4f}")
    
    # Save models
    joblib.dump(best_model, f'{MODEL_PATH}svr_model_{variant_id}.joblib')
    joblib.dump(scaler_X, f'{MODEL_PATH}scaler_X_{variant_id}.joblib')
    joblib.dump(features, f'{MODEL_PATH}features_{variant_id}.joblib')
    
    return {
        'y_test_actual': y_test_actual,
        'y_pred': y_pred,
        'rmse': rmse,
        'r2': r2
    }

def generate_prediction_chart(y_test_actual, y_pred, variant_id, rmse, r2):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual.reset_index(drop=True), label='Actual', color='green', alpha=0.7)
    plt.plot(y_pred, label='Prediction (SVR)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'SVR - {variant_id}\nRMSE: {rmse:.2f} | R²: {r2:.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    plt.savefig(f'{MODEL_PATH}plot_{variant_id}.png')
    plt.close()
    print(f"chart saved: {MODEL_PATH}plot_{variant_id}.png")

if __name__ == "__main__":
    # buat direktori baru
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    try:
        # ambil semua data dari db
        df_all = fetch_training_data()
        
        if df_all.empty:
            print("error: no training data found in database")
            exit()
            
    except Exception as e:
        print(f"error: failed to fetch data - {str(e)}")
        exit()
        
    unique_variants = df_all['product_variant_id'].unique()
    print(f"found {len(unique_variants)} unique product variants")
    
    for i, variant_id in enumerate(unique_variants):
        print(f"\n[{i+1}/{len(unique_variants)}] processing {variant_id}")
        df_variant = df_all[df_all['product_variant_id'] == variant_id].sort_values('date')
        try:
            # Train dan dapatkan metrics
            result = train_variant(df_variant, variant_id)
            
            if result:
                generate_prediction_chart(
                    result['y_test_actual'],
                    result['y_pred'],
                    variant_id,
                    result['rmse'],
                    result['r2']
                )
        except Exception as e:
            print(f"failed: {str(e)}")