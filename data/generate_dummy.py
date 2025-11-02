import csv
import random
from datetime import datetime, timedelta

# --- Konfigurasi ---
NAMA_FILE = 'data_historis.csv'
TANGGAL_AWAL = datetime(2024, 1, 1)
JUMLAH_HARI = 350  # 350 hari x 5 produk = 1750 baris data
MULTIPLIER_AKHIR_PEKAN = 1.8  # Penjualan 80% lebih tinggi di akhir pekan

# Konfigurasi produk [id_produk] -> {min_jual, max_jual}
PRODUK_CONFIG = {
    'A001': {'min': 10, 'max': 20},  # Laris
    'A002': {'min': 8, 'max': 15},   # Cukup Laris
    'B001': {'min': 3, 'max': 7},    # Sedang
    'B002': {'min': 1, 'max': 4},    # Kurang Laris
    'C001': {'min': 1, 'max': 3}     # Niche/Jarang
}

list_id_produk = list(PRODUK_CONFIG.keys())
data_rows = []

print(f"Memulai pembuatan data untuk file '{NAMA_FILE}'...")

# 1. Buat semua baris data dalam memori
for i in range(JUMLAH_HARI):
    current_date = TANGGAL_AWAL + timedelta(days=i)
    
    # Cek apakah akhir pekan (5 = Sabtu, 6 = Minggu)
    day_of_week = current_date.weekday()
    is_weekend = (day_of_week == 5 or day_of_week == 6)
    
    # Buat entri untuk setiap produk pada hari itu
    for id_produk in list_id_produk:
        config = PRODUK_CONFIG[id_produk]
        
        # Tentukan penjualan dasar
        penjualan_dasar = random.randint(config['min'], config['max'])
        
        # Terapkan tren akhir pekan
        if is_weekend:
            jumlah_terjual = int(penjualan_dasar * MULTIPLIER_AKHIR_PEKAN)
        else:
            jumlah_terjual = penjualan_dasar
            
        # Tambahkan sedikit "noise" agar tidak terlalu seragam
        noise = random.randint(-1, 2)
        jumlah_terjual = max(0, jumlah_terjual + noise) # Pastikan tidak negatif
        
        # Format tanggal DD-MM-YYYY
        tanggal_str = current_date.strftime('%d-%m-%Y')
        
        data_rows.append([tanggal_str, id_produk, jumlah_terjual])

# 2. Tulis semua data ke file CSV
try:
    with open(NAMA_FILE, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Tulis Header
        writer.writerow(['tanggal', 'id_produk', 'jumlah_terjual'])
        
        # Tulis Data
        writer.writerows(data_rows)
        
    print(f"Selesai! File '{NAMA_FILE}' berhasil dibuat dengan {len(data_rows)} baris data.")

except IOError as e:
    print(f"Error: Gagal menulis file. {e}")