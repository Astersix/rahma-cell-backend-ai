# 🧠 AI Service - Stock Prediction Engine

Layanan mikro (Microservice) khusus yang dibangun menggunakan Python untuk melakukan analisis prediktif stok barang bagi **CV Rahma Cell**. Layanan ini membantu pemilik toko menghindari kekosongan stok (*stockout*) atau penumpukan barang (*overstock*).

## 🤖 Teknologi & Model
* **Language:** Python 
* **Framework:** FastAPI 
* **Algoritma:** Support Vector Regression (SVR) 
* **Library:** Scikit-learn, Pandas, NumPy 

## 📊 Fungsi Utama
* **Training Model:** Melatih data historis penjualan untuk mengenali pola tren.
* **Stock Prediction:** Memberikan rekomendasi jumlah restock produk.
* **API Endpoint:** Menerima request data dari *Backend Main Service* dan mengembalikan hasil prediksi JSON.

## 🧪 Cara Menjalankan

1.  **Buat Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Windows: venv\Scripts\activate)
    ```
2.  **Install Library:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Jalankan Server AI:**
    ```bash
    uvicorn main:app --reload --port 8000
    ```

---
**Developed by Astersix Team**
