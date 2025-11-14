import joblib
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# konfigurasi
FILE_PATH = 'data/data_historis.csv'
# tiap produk memiliki tren tersendiri
PRODUCT_ID = 'A001'
# ambil data dari N hari untuk prediksi 1 hari ke depan
N_STEPS = 30
# pembagian data untuk latihan (contoh: 0.8, 0.2)
TEST_SIZE = 0.2
RANDOM_STATE = 42

def createSlidingWindow(data, n_steps):
    X_list, y_list = [], []
    
    if len(data) <= n_steps:
        print(f"Error: Data tidak cukup (panjang: {len(data)}) untuk window size ({n_steps}).")
        return np.array(X_list), np.array(y_list)

    for i in range(len(data) - n_steps):
        end_ix = i + n_steps
        seq_x = data[i:end_ix]
        seq_y = data[end_ix]
        X_list.append(seq_x)
        y_list.append(seq_y)
        
    return np.array(X_list), np.array(y_list)

def loadPreprocessData(file_path, produk_id, n_steps, test_size):
    try:
        # ambil dan proses data
        print(f"Membaca data dari '{file_path}'...")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{file_path}'.")
        print("Pastikan file 'data_historis.csv' ada di dalam folder 'data/'.")
        return None

    df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d-%m-%Y')
    
    # 2. saring data
    print(f"Melakukan filter untuk id_produk == '{produk_id}'...")
    df_produk = df[df['id_produk'] == produk_id].copy()
    df_produk.sort_values(by='tanggal', inplace=True)
    
    if df_produk.empty:
        print(f"Error: Tidak ditemukan data untuk produk '{produk_id}'.")
        return None

    time_series = df_produk['jumlah_terjual'].values
    print(f"Data time series asli ditemukan: {len(time_series)} baris.")

    # menerapkan Sliding Window
    print(f"Menerapkan sliding window dengan n_steps = {n_steps}...")
    X, y = createSlidingWindow(time_series, n_steps)
    
    if X.size == 0 or y.size == 0:
        print("Gagal membuat dataset X dan y. Proses dihentikan.")
        return None
        
    # reshape 'y' menjadi 2D agar bisa di-scale oleh
    # StandardScaler/MinMaxScaler (yang mengharapkan data 2D)
    y = y.reshape(-1, 1)

    # membagi Data (Train & Test)
    # Untuk time series, kita tidak boleh mengacak (shuffle=False).
    # Data tes harus merupakan data terbaru.
    print(f"Membagi data: {1-test_size:.0%} train, {test_size:.0%} test (tanpa shuffle)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=False,
        random_state=RANDOM_STATE
    )

    # scaling Data
    # Inisialisasi scaler terpisah untuk X dan y
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # fit scaler pada data train
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # menggunakan scaler yang sudah di-fit untuk men-transform data test
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # return semua data yang sudah diproses dan scaler_y
    # (scaler_y akan dibutuhkan nanti untuk inverse_transform prediksi)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

# eksekusi
if __name__ == "__main__":
    # menjalankan fungsi utama
    processed_data = loadPreprocessData(
        file_path=FILE_PATH,
        produk_id=PRODUCT_ID,
        n_steps=N_STEPS,
        test_size=TEST_SIZE
    )

    if processed_data:
        # memecah data
        X_train_s, X_test_s, y_train_s, y_test_s, scaler_X, scaler_y = processed_data
        
        print("shapes")
        print(f"Bentuk X_train (fitur, scaled): \t{X_train_s.shape}")
        print(f"Bentuk y_train (target, scaled): \t{y_train_s.shape}")
        print(f"Bentuk X_test (fitur, scaled): \t\t{X_test_s.shape}")
        print(f"Bentuk y_test (target, scaled): \t{y_test_s.shape}")
        
        # pakai rbf untuk tren non-linear
        model = SVR(kernel='rbf', C=1.0)
        # latih model pada data training (scaled)
        # pakai ravel untuk ubah data kembali ke 1D untuk memenuhi .fit()
        model.fit(X_train_s, y_train_s.ravel())
        
        # prediksi (scaled) pada data test
        y_pred_s = model.predict(X_test_s)
        
        # ubah scaled menjadi real (sebelum scaling)
        y_pred_r = scaler_y.inverse_transform(y_pred_s.reshape(-1,1))
        y_test_r = scaler_y.inverse_transform(y_test_s)
        
        # hitung selisih antara prediksi dan test
        mse = mean_squared_error(y_test_r, y_pred_r)
        rmse = np.sqrt(mse)
        print(f"skor mse: {mse:.4f}")
        print(f"skor rmse: {rmse:.4f}")
        
        # buat grafik
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_r, label='data aktual (y_test)', color='green', marker='o', markersize=5)
        plt.plot(y_pred_r, label='data prediksi (y_pred)', color='red', linestyle='--', marker='x', markersize=5)
        plt.title(f"perbadingan prediksi dan aktual (produk: {PRODUCT_ID})")
        plt.xlabel("Hari")
        plt.ylabel("Terjual")
        
        Y_STEP = 5
        X_STEP = 7
        max_y_val = max(np.max(y_test_r), np.max(y_pred_r))
        max_x_val = len(y_test_r)
        y_ticks = np.arange(0, max_y_val + Y_STEP, Y_STEP)
        x_ticks = np.arange(0, max_x_val + X_STEP, X_STEP)

        plt.yticks(y_ticks)
        plt.xticks(x_ticks)
        
        plt.legend()
        plt.grid(True)
        
        # simpan grafik
        plot_filename = f'grafik_prediksi_{PRODUCT_ID}.png'
        plt.savefig(plot_filename)
        plt.close()

        # simpan model dan scaler untuk api
        os.makedirs('models', exist_ok=True)

        joblib.dump(model, f"models/svr_model_{PRODUCT_ID}.joblib")
        joblib.dump(scaler_X, f"models/scaler_X_{PRODUCT_ID}.joblib")
        joblib.dump(scaler_y, f"models/scaler_y_{PRODUCT_ID}.joblib")
        print("model dan scaler berhasil disimpan")
        