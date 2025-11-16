import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("--- Memulai Script Training Model (SVR + GridSearchCV) ---")

FILE_PATH = 'data/dummy_features_3000.csv'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_FILENAME = 'svr_model.joblib'
SCALER_X_FILENAME = 'scaler_X.joblib' # Kita perlu menyimpan scaler

# --- 2. Memuat Data Fitur (X) ---
try:
    X_features = pd.read_csv(FILE_PATH)
    print(f"Berhasil memuat data fitur dari '{FILE_PATH}'.")
    print(f"Bentuk data fitur (X) awal: {X_features.shape}")
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di '{FILE_PATH}'. Hentikan script.")
    exit()

# --- 3. [PENTING] Membuat Target (y) ---
# Kita buat ulang target (y) yang ingin kita prediksi.
print("Membuat ulang target 'penjualan_hari_ini' (y)...")
base_sales = 20
price_effect = - (X_features['product.price'] / 15)
weekend_effect = 25 * X_features['is_weekend']
cart_effect = 0.1 * X_features['total_items_in_carts']
noise = np.random.normal(0, 5, len(X_features)) # Target kita punya noise 5
simulated_sales = base_sales + price_effect + weekend_effect + cart_effect + noise
y_target = np.maximum(0, simulated_sales).round().astype(int)


# --- 4. Feature Engineering (Encoding) ---
print("Melakukan One-Hot Encoding pada fitur kategorikal...")
categorical_cols = ['product.category_id', 'day_of_week', 'month']
X_encoded = pd.get_dummies(X_features, columns=categorical_cols, drop_first=True)
print(f"Bentuk data (X) setelah encoding: {X_encoded.shape}")

# Simpan nama-nama kolom untuk nanti
COLUMN_NAMES = X_encoded.columns.tolist()


# --- 5. Membagi Data (Train & Test) ---
# PENTING: Untuk data time series, kita TIDAK boleh mengacak (shuffle=False).
print("Membagi data (tanpa shuffle)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_target, 
    test_size=TEST_SIZE, 
    shuffle=False, 
    random_state=RANDOM_STATE
)

# --- 6. SCALING DATA (Wajib untuk SVR) ---
# Kita 'fit' (belajar) HANYA pada data training
# dan 'transform' (menerapkan) pada data train dan test.
print("Melakukan scaling data (StandardScaler)...")
scaler_X = StandardScaler()

# Fit dan transform X_train
X_train_scaled = scaler_X.fit_transform(X_train)

# HANYA transform X_test (menggunakan skala dari X_train)
X_test_scaled = scaler_X.transform(X_test)

print(f"Bentuk X_train_scaled: {X_train_scaled.shape}")


# --- 7. Melatih Model (SVR dengan GridSearchCV) ---
print("Memulai Hyperparameter Tuning SVR dengan GridSearchCV...")
print("Ini akan memakan waktu (n_jobs=1 untuk menghindari bug)...")

# Tentukan SVR dasar
svr_base = SVR(kernel='rbf')

# Tentukan grid parameter. Mulailah dari yang kecil.
# Jika RMSE masih tinggi, coba 'C': [100, 1000] atau 'gamma': [0.1, 1]
param_grid = {
    'C': [1, 10],         # Coba C=1 dan C=10
    'gamma': [0.1, 0.01]  # Coba gamma=0.1 dan gamma=0.01
}

grid_search = GridSearchCV(
    estimator=svr_base,
    param_grid=param_grid,
    cv=3,                          # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    n_jobs=1,                      # Ganti ke -1 jika Anda yakin aman
    verbose=2                      # Tampilkan log progres
)

# Latih GridSearchCV pada data yang sudah di-scale
# y_train tidak perlu di-scale, tapi .ravel() direkomendasikan
grid_search.fit(X_train_scaled, y_train.ravel())

# Ambil model terbaik
best_model = grid_search.best_estimator_

print("\n--- Tuning Selesai ---")
print(f"Parameter terbaik ditemukan: {grid_search.best_params_}")


# --- 8. Evaluasi Model ---
print("\n--- Evaluasi Model ---")
# Lakukan prediksi pada X_test_scaled
y_pred = best_model.predict(X_test_scaled)

# Hitung error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"HASIL AKHIR (TEST SET):")
print(f"  Mean Squared Error (MSE): {mse:.4f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")

# Target kita dibuat dengan noise 5. Jika RMSE Anda mendekati 5.0,
# itu artinya model Anda sangat akurat.

print("Membuat plot perbandingan performa model...")
plt.figure(figsize=(15, 6))

# y_test adalah Pandas Series, y_pred adalah Numpy array. 
# Kita plot keduanya.
plt.plot(np.array(y_test), label='Data Aktual (y_test)', color='blue', marker='.', alpha=0.7)
plt.plot(y_pred, label='Data Prediksi (y_pred)', color='red', linestyle='--', alpha=0.7)

plt.title(f'Performa Model SVR (RMSE: {rmse:.4f})')
plt.xlabel('Time Step (pada Data Tes)')
plt.ylabel('Jumlah Terjual')

# Mengatur kelipatan sumbu Y agar lebih rapi
try:
    max_val = max(np.max(y_test), np.max(y_pred))
    Y_KELIPATAN = 10 # Anda bisa ubah kelipatan ini
    y_ticks = np.arange(0, max_val + Y_KELIPATAN, Y_KELIPATAN)
    plt.yticks(y_ticks)
except ValueError:
    # Terjadi jika y_test kosong (jarang terjadi)
    print("Gagal mengatur Y-Ticks.")
    
plt.legend()
plt.grid(True)

plot_filename = 'plot_performa_svr.png'
plt.savefig(plot_filename) # Simpan plot
plt.close()
print(f"Plot performa berhasil disimpan sebagai '{plot_filename}'")


# --- 9. Menyimpan Model & Scaler ---
print("\n--- Menyimpan Model dan Scaler ---")
joblib.dump(best_model, MODEL_FILENAME)
print(f"Model berhasil disimpan sebagai '{MODEL_FILENAME}'")

joblib.dump(scaler_X, SCALER_X_FILENAME)
print(f"Scaler X berhasil disimpan sebagai '{SCALER_X_FILENAME}'")

# Simpan juga daftar nama kolom
joblib.dump(COLUMN_NAMES, 'column_names.joblib')
print(f"Nama kolom disimpan di 'column_names.joblib'")

print("\n--- Script Selesai ---")