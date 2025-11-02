import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- KONFIGURASI ---
FILE_PATH = '../data/data_historis.csv' # Path relatif dari src/ ke data/
PRODUK_ID = 'A001'                     # Produk yang akan di-modelkan
N_STEPS = 7                            # Ukuran Window: 7 hari data historis
TEST_SIZE = 0.2                        # 20% data untuk validasi/testing
RANDOM_STATE = 42                      # Untuk reproduktibilitas (opsional)

def create_sliding_window(data, n_steps):
    """
    Mengubah array 1D time series menjadi dataset supervised learning
    dengan format sliding window.
    """
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

def load_and_preprocess_data(file_path, produk_id, n_steps, test_size):
    """
    Fungsi utama untuk memuat, memproses, membagi, 
    dan menskalakan data time series.
    """
    try:
        # 1. Muat dan Proses Data
        print(f"Membaca data dari '{file_path}'...")
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di '{file_path}'.")
        print("Pastikan file 'data_historis.csv' ada di dalam folder 'data/'.")
        return None

    df['tanggal'] = pd.to_datetime(df['tanggal'], format='%d-%m-%Y')
    
    # 2. Filter dan Siapkan Data
    print(f"Melakukan filter untuk id_produk == '{produk_id}'...")
    df_produk = df[df['id_produk'] == produk_id].copy()
    df_produk.sort_values(by='tanggal', inplace=True)
    
    if df_produk.empty:
        print(f"Error: Tidak ditemukan data untuk produk '{produk_id}'.")
        return None

    time_series = df_produk['jumlah_terjual'].values
    print(f"Data time series asli ditemukan: {len(time_series)} baris.")

    # 3. Terapkan Sliding Window
    print(f"Menerapkan sliding window dengan n_steps = {n_steps}...")
    X, y = create_sliding_window(time_series, n_steps)
    
    if X.size == 0 or y.size == 0:
        print("Gagal membuat dataset X dan y. Proses dihentikan.")
        return None
        
    # PENTING: Reshape 'y' menjadi 2D agar bisa di-scale oleh
    # StandardScaler/MinMaxScaler (yang mengharapkan data 2D)
    y = y.reshape(-1, 1)

    # 4. Bagi Data (Train & Test)
    # PENTING: Untuk time series, kita TIDAK boleh mengacak (shuffle=False).
    # Data tes harus merupakan data terbaru.
    print(f"Membagi data: {1-test_size:.0%} train, {test_size:.0%} test (tanpa shuffle)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=False, # <-- SANGAT PENTING UNTUK TIME SERIES
        random_state=RANDOM_STATE # (random_state tidak berpengaruh jika shuffle=False)
    )

    # 5. Scaling Data
    print("Melakukan scaling data (MinMaxScaler)...")
    # Inisialisasi scaler terpisah untuk X dan y
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Fit scaler HANYA pada data TRAIN
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Gunakan scaler yang sudah di-fit untuk men-transform data TEST
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    print("Proses Selesai.")
    
    # Mengembalikan semua data yang sudah diproses dan scaler_y
    # (scaler_y akan dibutuhkan nanti untuk inverse_transform prediksi)
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_y

# --- EKSEKUSI SCRIPT ---
if __name__ == "__main__":
    print("--- Memulai Pemrosesan Data untuk Training ---")
    
    # Jalankan fungsi utama
    processed_data = load_and_preprocess_data(
        file_path=FILE_PATH,
        produk_id=PRODUK_ID,
        n_steps=N_STEPS,
        test_size=TEST_SIZE
    )

    if processed_data:
        # Unpack data
        X_train_s, X_test_s, y_train_s, y_test_s, _ = processed_data
        
        print("\n--- HASIL AKHIR (SHAPES) ---")
        print(f"Bentuk X_train (fitur, scaled): \t{X_train_s.shape}")
        print(f"Bentuk y_train (target, scaled): \t{y_train_s.shape}")
        print(f"Bentuk X_test (fitur, scaled): \t\t{X_test_s.shape}")
        print(f"Bentuk y_test (target, scaled): \t{y_test_s.shape}")
        
        print("\nContoh 1 baris X_train_scaled:")
        print(X_train_s[0])
        print("\nContoh 1 baris y_train_scaled:")
        print(y_train_s[0])