import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ulid

# konfigurasi
OUTPUT_FILE = 'data/data_penjualan_dummy.csv'
N_ROWS = 100000
N_VARIANTS = 100

def generate_data():
    print(f"generating {N_ROWS} rows of dummy data with ulid...")

    # setup tanggal
    end_date = datetime.now()
    dates = [end_date - timedelta(days=x) for x in range(N_ROWS)]
    dates.reverse()

    df = pd.DataFrame()
    df['date'] = dates
    
    # --- BAGIAN GENERATE ULID ---
    variants_db = {}
    
    print("generated variant ids (ulid):")
    for i in range(N_VARIANTS):
        # PERBAIKAN DI SINI:
        # Gunakan ulid.ULID() untuk membuat instance baru
        u_id = str(ulid.ULID()) 
        
        # tentukan karakteristik
        base_price = np.random.randint(50, 150) * 1000 
        popularity = np.random.randint(0, 20) 
        
        variants_db[u_id] = {
            'base_price': base_price,
            'popularity': popularity
        }
        print(f"- {u_id} (price: {base_price}, popularity: {popularity})")

    available_ulids = list(variants_db.keys())

    # assign variant id secara acak
    df['product_variant_id'] = np.random.choice(available_ulids, N_ROWS)
    
    df['category_id'] = 101 
    
    # helper functions
    def get_base_price(uid):
        return variants_db[uid]['base_price']
    
    def get_popularity(uid):
        return variants_db[uid]['popularity']

    # map karakteristik ke dataframe
    df['base_price_ref'] = df['product_variant_id'].apply(get_base_price)
    df['popularity_ref'] = df['product_variant_id'].apply(get_popularity)
    
    # simulasi harga
    df['price'] = df['base_price_ref'] + np.random.normal(0, 3000, N_ROWS)
    df['price'] = df['price'].round(-2) 

    # simulasi penjualan
    df['day_of_week'] = df['date'].dt.dayofweek
    weekend_effect = np.where(df['day_of_week'] >= 5, 15, 0)
    
    price_sensitivity = (df['base_price_ref'] - df['price']) / 1000
    
    time_index = np.arange(N_ROWS)
    seasonality = 8 * np.sin(time_index * (2 * np.pi / 30))

    base_sales = 20
    noise = np.random.normal(0, 5, N_ROWS)
    
    simulated_qty = base_sales + df['popularity_ref'] + weekend_effect + price_sensitivity + seasonality + noise
    
    df['quantity'] = np.maximum(0, simulated_qty).round().astype(int)

    # bersihkan dan simpan
    df = df.drop(columns=['base_price_ref', 'popularity_ref'])

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"data berhasil disimpan ke {OUTPUT_FILE}")
    print("preview data:")
    print(df.head())

if __name__ == "__main__":
    generate_data()