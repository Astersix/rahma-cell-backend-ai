# 📦 Rahma Cell - AI Backend

> **AI-powered Inventory Stock Prediction API** menggunakan Support Vector Regression (SVR) untuk prediksi stok inventori yang akurat.

## 🎯 Fitur Utama

- ✅ **Prediksi Stok Cerdas** - Algoritma SVR dengan hyperparameter tuning
- ✅ **Database Integration** - Direct fetch dari MySQL untuk retraining
- ✅ **RESTful API** - FastAPI dengan automatic documentation
- ✅ **Model Persistence** - Automatic model saving & loading
- ✅ **Feature Engineering** - Time series features (lag, rolling stats)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan (Windows)
.\venv\Scripts\activate

# Aktifkan (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Database

Buat file `.env` di root directory:

```env
DATABASE_URL="mysql+pymysql://user:password@localhost:3306/dbname"
PORT=8000
```

> **Tip:** Copy dari `.env.example` dan sesuaikan dengan konfigurasi Anda.

**Option A: Training dari Database (Recommended)**

```bash
# Train model dengan data dari database
python src/train.py
```

**Option B: Retraining Model yang Sudah Ada**

```bash
# Retrain semua model dengan data terbaru
python src/retrain.py
```

### 5. Run API Server

```bash
uvicorn src.main:app --reload
```

API akan berjalan di `http://localhost:8000`

📚 **Documentation:** `http://localhost:8000/docs`

## 📖 API Usage

### Predict Stock Endpoint

```bash
POST /predict-stock
```

**Request Body:**
```json
{
  "product_variant_id": "PROD001",
  "price": 15000,
  "category_id": 1,
  "last_transaction_date": "2025-11-28",
  "sales_history_30_days": [5, 7, 6, 8, 10, ...],
  "current_stock": 50  // Optional: defaults to 0, will be fetched from database if not provided
}
```

**Response:**
```json
{
  "product_variant_id": "PROD001",
  "prediction_period": "next 7 days",
  "daily_predictions": [8.5, 9.2, 7.8, 10.1, 9.5, 11.2, 8.9],
  "total_restock_recommended": 8  // Additional stock needed = max(0, forecast - current_stock)
}
```

> **Note:** `total_restock_recommended` represents the additional stock needed beyond current inventory. If current stock is sufficient, this value will be 0.

**Rate Limiting:** 30 requests per minute per IP address

## 🏗️ Project Structure

```
rahma-cell-backend-ai/
├── src/
│   ├── main.py          # FastAPI application & prediction endpoint
│   ├── train.py         # Initial training script from database
│   ├── retrain.py       # Retraining script (with file locking)
│   └── test.py          # Stress testing script
├── data/
│   └── generate_dummy.py # Dummy data generator (deprecated)
├── models/              # Saved models, scalers, features (.joblib)
├── requirements.txt
├── .env                 # Environment variables (gitignored)
└── README.md
```

## 🧠 Model Details

**Algorithm:** Support Vector Regression (SVR) with GridSearchCV hyperparameter tuning

**Features:** (19 total - dinamically selected based on available data)


*Temporal Features:*

- `day_of_week` - Hari dalam seminggu (0-6)
- `is_weekend` - Binary weekend indicator (1 = weekend, 0 = weekday)
- `day` - Tanggal dalam bulan (1-31)
- `trend` - Indeks waktu linear

*Price Feature:*

- `price` - Harga produk (rata-rata untuk hari tersebut)


*Lag Features:*

- `lag_1` to `lag_5` - Penjualan 1-5 hari sebelumnya

*Rolling Statistics:*
- `rolling_mean_3` - Mean 3 hari terakhir
- `rolling_std_3` - Standard deviation 3 hari
- `rolling_min_3` - Minimum 3 hari terakhir
- `rolling_max_3` - Maximum 3 hari terakhir

*Momentum Features:*

- `momentum` - Perubahan penjualan dari hari sebelumnya
- `momentum_3` - Perubahan penjualan dalam 3 hari

**Target:** `quantity` (jumlah terjual per hari - log transformed)

## 🔧 Advanced Usage

### Custom Training Configuration

Edit `src/train.py`:
```python
MIN_DATA_POINTS = 30          # Minimum data points per variant
# Script trains all variants from database automatically
```

### Retraining Schedule

Setup scheduled retraining dengan cron (Linux/Mac) atau Task Scheduler (Windows):

```bash
# Contoh: retrain setiap minggu
0 2 * * 0 cd /path/to/project && python src/retrain.py
```

### Stress Testing

Test performa API dengan `test.py`:

```bash
python src/test.py
```

Script akan melakukan 1000 request berturut-turut dan menampilkan:
- Response time statistics (avg, min, max, p95)
- Error rate
- Grafik response time


## 📊 Model Performance

Model di-evaluasi menggunakan metrics:
- **RMSE** (Root Mean Squared Error)
- **R²** Score

Prediction charts disimpan di `models/plot_{variant_id}.png`

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)

---

**Made with ❤️ by Astersix**
