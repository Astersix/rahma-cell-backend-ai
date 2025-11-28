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

### 4. Training Model

**Option A: Training dari CSV (Development)**
```bash
# Generate dummy data
python data/generate_dummy.py

# Train model
python src/train.py
```

**Option B: Retraining dari Database (Production)**
```bash
# Retrain dengan data real dari database
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
  "sales_history_30_days": [5, 7, 6, 8, 10, ...]
}
```

**Response:**
```json
{
  "product_variant_id": "PROD001",
  "prediction_period": "next 7 days",
  "daily_predictions": [8.5, 9.2, 7.8, ...],
  "total_restock_recommended": 58
}
```

## 🏗️ Project Structure

```
rahma-cell-backend-ai/
├── src/
│   ├── main.py          # FastAPI application
│   ├── train.py         # Initial training script
│   └── retrain.py       # Retraining from database
├── data/
│   └── generate_dummy.py
├── models/              # Saved models (.joblib)
├── requirements.txt
├── .env                 # Environment variables (gitignored)
└── README.md
```

## 🧠 Model Details

**Algorithm:** Support Vector Regression (SVR)

**Features:**
- `price` - Harga produk
- `day_of_week` - Hari dalam seminggu (0-6)
- `is_weekend` - Binary weekend indicator
- `lag_1` - Penjualan 1 hari sebelumnya
- `lag_7` - Penjualan 7 hari sebelumnya
- `rolling_mean_7` - Mean 7 hari terakhir
- `rolling_std_7` - Standard deviation 7 hari

**Target:** `quantity` (jumlah terjual)

## 🔧 Advanced Usage

### Custom Training Configuration

Edit `src/train.py`:
```python
MIN_DATA_POINTS = 50          # Minimum data points per variant
MAX_VARIANTS_TO_TRAIN = None  # None = semua variant
```

### Retraining Schedule

Setup scheduled retraining dengan cron (Linux/Mac) atau Task Scheduler (Windows):

```bash
# Contoh: retrain setiap minggu
0 2 * * 0 cd /path/to/project && python src/retrain.py
```

## 📊 Model Performance

Model di-evaluasi menggunakan metrics:
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** Score

Learning curves disimpan di `models/learning_curve_{variant_id}.png`

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)

---

**Made with ❤️ by Astersix**
