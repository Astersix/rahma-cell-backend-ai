import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# konfigurasi
FILE_PATH = 'data/data_penjualan_dummy.csv'
MODEL_PATH = 'models/'
MIN_DATA_POINTS = 50 

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    return df

def feature_engineering(df):
    # fitur waktu
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # fitur lag dan rolling
    df['lag_1'] = df['quantity'].shift(1)
    df['lag_7'] = df['quantity'].shift(7)
    df['rolling_mean_7'] = df['quantity'].shift(1).rolling(window=7).mean()
    df['rolling_std_7'] = df['quantity'].shift(1).rolling(window=7).std()
    
    df = df.dropna()
    return df

def train_for_variant(df_variant, variant_id):
    print(f"\nprocessing variant: {variant_id}")
    
    # cek apakah data cukup
    if len(df_variant) < MIN_DATA_POINTS:
        print(f"skip: insufficient data ({len(df_variant)} rows)")
        return

    df_processed = feature_engineering(df_variant.copy())
    
    # cek lagi setelah dropna
    if len(df_processed) < 30:
        print("skip: insufficient data after processing")
        return

    # definisi fitur dan target
    features = ['price', 'category_id', 'day_of_week', 'day_of_month', 'is_weekend', 
                'lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7']
    target = 'quantity'
    
    X = df_processed[features]
    y = df_processed[target]
    
    # bagi data training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # penskalaan (scaling)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    X_test_scaled = scaler_X.transform(X_test)
    
    # training menggunakan gridsearch
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.1],
        'epsilon': [0.1, 0.2]
    }
    
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train_scaled.ravel())
    
    best_model = grid_search.best_estimator_
    
    # evaluasi model
    y_pred_scaled = best_model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # tampilkan skor error ke terminal
    print(f"success. mse: {mse:.4f}, rmse: {rmse:.4f}")

    # pembuatan grafik (plotting)
    plt.figure(figsize=(10, 5))
    
    # plot data aktual (hijau)
    plt.plot(y_test.reset_index(drop=True), label='Actual', color='green', alpha=0.7)
    
    # plot data prediksi (merah)
    plt.plot(y_pred, label='Prediction', color='red', linestyle='--', alpha=0.8)
    
    # judul grafik dengan skor error
    plt.title(f'Prediction vs Actual - {variant_id}\nMSE: {mse:.2f}, RMSE: {rmse:.2f}')
    plt.xlabel('Data Point (Test Set)')
    plt.ylabel('Quantity')
    
    plt.legend() # tampilkan legenda
    plt.grid(True, alpha=0.3)
    
    os.makedirs(MODEL_PATH, exist_ok=True)
    plt.savefig(f'{MODEL_PATH}plot_{variant_id}.png')
    plt.close()
    
    # simpan model dan scaler
    joblib.dump(best_model, f'{MODEL_PATH}svr_model_{variant_id}.joblib')
    joblib.dump(scaler_X, f'{MODEL_PATH}scaler_X_{variant_id}.joblib')
    joblib.dump(scaler_y, f'{MODEL_PATH}scaler_y_{variant_id}.joblib')

if __name__ == "__main__":
    print("--- starting batch training ---")
    
    # muat dataset utama
    try:
        df_all = load_data(FILE_PATH)
    except FileNotFoundError:
        print("error: csv file not found")
        exit()
        
    # ambil semua varian unik
    unique_variants = df_all['product_variant_id'].unique()
    print(f"total variants: {len(unique_variants)}")
    
    # loop untuk setiap varian
    for i, variant_id in enumerate(unique_variants):
        print(f"\n[{i+1}/{len(unique_variants)}] checking variant {variant_id}")
        df_variant = df_all[df_all['product_variant_id'] == variant_id].sort_values('date')
        
        try:
            train_for_variant(df_variant, variant_id)
        except Exception as e:
            print(f"failed: {str(e)}")

    print("\n--- finished ---")