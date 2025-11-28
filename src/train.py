import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# configuration
FILE_PATH = 'data/data_penjualan_dummy.csv'
MODEL_PATH = 'models/'
MIN_DATA_POINTS = 50 

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def feature_engineering(df):
    # time features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    df['lag_1'] = df['quantity'].shift(1)
    df['lag_7'] = df['quantity'].shift(7)
    df['rolling_mean_7'] = df['quantity'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['quantity'].shift(1).rolling(window=7).std()
    
    df = df.dropna()
    return df

def train_variant(df_variant, variant_id):
    print(f"\nprocessing variant: {variant_id}")
    
    if len(df_variant) < MIN_DATA_POINTS:
        print(f"skip: insufficient data ({len(df_variant)} rows)")
        return None

    df_processed = feature_engineering(df_variant.copy())
    
    if len(df_processed) < 30:
        print("skip: insufficient data after processing")
        return None

    features = ['price', 'day_of_week', 'is_weekend', 
                'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
    target = 'quantity'
    
    X = df_processed[features]
    y = df_processed[target]
    
    y_log = np.log1p(y)
    
    # split data
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, shuffle=False)
    
    # scaling features (wajib untuk svr)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # training
    param_grid = {
        'C': [10, 100, 1000, 5000],
        'gamma': ['scale', 0.1, 0.01],
        'epsilon': [0.01, 0.1]
    }
    
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    
    # train menggunakan target log
    grid_search.fit(X_train_scaled, y_train_log)
    
    best_model = grid_search.best_estimator_
    print(f"best params: {grid_search.best_params_}")
    
    # evaluasi
    y_pred_log = best_model.predict(X_test_scaled)
    
    y_pred = np.expm1(y_pred_log)
    
    # kembalikan data test asli (karena y_test_log masih log)
    y_test_actual = np.expm1(y_test_log)
    
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"success. mse: {mse:.4f}, rmse: {rmse:.4f}")
    
    # save models
    joblib.dump(best_model, f'{MODEL_PATH}svr_model_{variant_id}.joblib')
    joblib.dump(scaler_X, f'{MODEL_PATH}scaler_X_{variant_id}.joblib')
    joblib.dump(features, f'{MODEL_PATH}features_{variant_id}.joblib')
    
    # return data untuk chart
    return {
        'y_test_actual': y_test_actual,
        'y_pred': y_pred,
        'rmse': rmse
    }

def generate_prediction_chart(y_test_actual, y_pred, variant_id, rmse):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test_actual.reset_index(drop=True), label='Actual', color='green', alpha=0.7)
    plt.plot(y_pred, label='Prediction (SVR)', color='red', linestyle='--', alpha=0.8)
    plt.title(f'SVR Optimized - {variant_id}\\nRMSE: {rmse:.2f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    plt.savefig(f'{MODEL_PATH}plot_{variant_id}.png')
    plt.close()
    print(f"chart saved: {MODEL_PATH}plot_{variant_id}.png")

if __name__ == "__main__":
    print("--- starting batch training (svr optimized) ---")
    try:
        df_all = load_data(FILE_PATH)
    except FileNotFoundError:
        print("error: csv file not found")
        exit()
        
    unique_variants = df_all['product_variant_id'].unique()
    for i, variant_id in enumerate(unique_variants):
        print(f"\\n[{i+1}/{len(unique_variants)}] processing {variant_id}")
        df_variant = df_all[df_all['product_variant_id'] == variant_id].sort_values('date')
        try:
            # Train dan dapatkan metrics
            result = train_variant(df_variant, variant_id)
            
            if result:
                generate_prediction_chart(
                    result['y_test_actual'],
                    result['y_pred'],
                    variant_id,
                    result['rmse']
                )
        except Exception as e:
            print(f"failed: {str(e)}")