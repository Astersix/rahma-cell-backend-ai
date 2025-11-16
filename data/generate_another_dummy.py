import pandas as pd
import numpy as np

# Set the number of rows
N_ROWS = 3000

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Generate Independent & Product Features ---

# Create a DataFrame
df = pd.DataFrame(index=range(N_ROWS))

# `day_of_week` (0=Senin, 6=Minggu)
df['day_of_week'] = np.random.randint(0, 7, N_ROWS)
# `is_weekend` (1 if Sabtu/Minggu, 0 otherwise)
df['is_weekend'] = np.where(df['day_of_week'] >= 5, 1, 0)
# `month` (1-12)
df['month'] = np.random.randint(1, 13, N_ROWS)

# `product.category_id` (e.g., 3 categories: 10, 20, 30)
df['product.category_id'] = np.random.choice([10, 20, 30], N_ROWS, p=[0.5, 0.3, 0.2])

# `product.price` (dependent on category)
price_map = {10: 25.0, 20: 80.0, 30: 150.0}
# Base price from map + some random "noise"
df['product.price'] = df['product.category_id'].map(price_map) + np.random.normal(0, 5, N_ROWS)
df['product.price'] = np.maximum(5, df['product.price']) # Ensure no negative/too-low prices

# `product_age` (in days)
df['product_age'] = np.random.randint(30, 1000, N_ROWS)

# `product.stock_quantity`
df['product.stock_quantity'] = np.random.randint(0, 500, N_ROWS)

# --- 2. Generate Cart Features ---

# `total_items_in_carts`
df['total_items_in_carts'] = np.random.randint(0, 150, N_ROWS)
# `total_users_with_item_in_cart`
df['total_users_with_item_in_cart'] = np.random.randint(0, df['total_items_in_carts'] + 1)
# Ensure users <= items
df['total_users_with_item_in_cart'] = np.minimum(df['total_users_with_item_in_cart'], df['total_items_in_carts'])
df.loc[df['total_items_in_carts'] == 0, 'total_users_with_item_in_cart'] = 0


# --- 3. Simulate Sales (to create realistic lag features) ---
# We create a "penjualan_hari_ini" column to base our lag features on.
# This logic makes the features interdependent and realistic.

# Base sales
base_sales = 20
# Effects from features
price_effect = - (df['product.price'] / 15)
weekend_effect = 25 * df['is_weekend']
month_effect = 10 * np.sin((df['month'] - 3) * (2 * np.pi / 12)) # Seasonal effect (peak around May/June)
cart_effect = 0.1 * df['total_items_in_carts']
stock_out_effect = -20 * (df['product.stock_quantity'] == 0) # Sales drop if stock is 0
noise = np.random.normal(0, 5, N_ROWS)

# Combine effects to get simulated sales
simulated_sales = base_sales + price_effect + weekend_effect + month_effect + cart_effect + stock_out_effect + noise
# Ensure sales are not negative and are integers
df['penjualan_hari_ini'] = np.maximum(0, simulated_sales).round().astype(int)


# --- 4. Derive Lag/Rolling Features (The user's requested features) ---
# These are calculated from the simulated 'penjualan_hari_ini'

# `penjualan_kemarin`
df['penjualan_kemarin'] = df['penjualan_hari_ini'].shift(1).fillna(0)

# `penjualan_7_hari_lalu`
df['penjualan_7_hari_lalu'] = df['penjualan_hari_ini'].shift(7).fillna(0)

# `rata_rata_penjualan_7_hari` (based on 7 days ending yesterday)
df['rata_rata_penjualan_7_hari'] = df['penjualan_hari_ini'].shift(1).rolling(window=7).mean().fillna(0)

# `std_dev_penjualan_7_hari` (based on 7 days ending yesterday)
df['std_dev_penjualan_7_hari'] = df['penjualan_hari_ini'].shift(1).rolling(window=7).std().fillna(0)


# --- 5. Finalize and Save ---

# Define the list of 13 features the user requested
final_features = [
    'product.price',
    'product.category_id',
    'product.stock_quantity',
    'day_of_week',
    'month',
    'is_weekend',
    'product_age',
    'penjualan_kemarin',
    'penjualan_7_hari_lalu',
    'rata_rata_penjualan_7_hari',
    'std_dev_penjualan_7_hari',
    'total_items_in_carts',
    'total_users_with_item_in_cart'
]

# Create the final DataFrame
df_final = df[final_features]

# Save to CSV
output_filename = 'dummy_features_3000.csv'
df_final.to_csv(output_filename, index=False)

print(f"Successfully generated 3000 rows of dummy data with 13 features.")
print(f"File saved as: {output_filename}")
print("\n--- Head of the dummy data (first 5 rows) ---")
print(df_final.head())