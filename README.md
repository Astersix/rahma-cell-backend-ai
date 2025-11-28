# rahma-cell-backend-ai

rahma-cell-backend-ai adalah API model AI yang digunakan untuk memprediksi stok inventori barang dengan algoritma SVR.

## Setup

Buat virtual environment.

```bash
python -m venv venv
```

Aktifkan virtual environment.

```bash
.\venv\Scripts\activate.bat
```

Install library yang diperlukan.

```bash
pip install -r requirements.txt
```

Generate dummy data (data aktual lebih disarankan).

```bash
python  /data/generate_dummy.py
```

Latih model.

```bash
python  /src/train.py
```

Jalankan server API.

```bash
uvicorn src.main:app --reload
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
