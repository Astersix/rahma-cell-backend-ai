import requests
import time
import os
import statistics
import matplotlib.pyplot as plt

# konfigurasi
URL = "http://127.0.0.1:8000/predict-stock"
MODEL_DIR = "models/"
TOTAL_REQUESTS = 5

def get_test_variant_id():
    # ambil salah satu model yang ada di folder models
    if not os.path.exists(MODEL_DIR):
        return None
    for f in os.listdir(MODEL_DIR):
        if f.startswith("svr_model_") and f.endswith(".joblib"):
            return f.replace("svr_model_", "").replace(".joblib", "")
    return None

def run_stress_test():
    print("memulai stress testing")
    
    variant_id = get_test_variant_id()
    if not variant_id:
        print("error: tidak ditemukan model untuk ditest")
        return

    print(f"target variant: {variant_id}")
    print(f"jumlah request: {TOTAL_REQUESTS}")

    # dummy payload (data acak 30 hari)
    payload = {
        "product_variant_id": variant_id,
        "price": 100000.0,
        "category_id": 101,
        "last_transaction_date": "2025-11-22",
        "sales_history_30_days": [25] * 30
    }
    
    response_times = []
    errors = 0

    print("\nmemulai loop request...")
    
    # loop request berturut-turut
    for i in range(TOTAL_REQUESTS):
        start_time = time.time()
        
        try:
            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                end_time = time.time()
                duration = (end_time - start_time) * 1000 # konversi ke ms
                response_times.append(duration)
                
                # print status setiap 10 request agar tidak spam konsol
                if (i + 1) % 10 == 0:
                    print(f"req {i+1}/{TOTAL_REQUESTS} - status: {response.status_code} - time: {duration:.2f} ms")
            else:
                errors += 1
                print(f"req {i+1} gagal. status: {response.status_code}")
                
        except Exception as e:
            errors += 1
            print(f"req {i+1} error koneksi: {e}")

    # kalkulasi hasil
    if response_times:
        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        p95_time = statistics.quantiles(response_times, n=20)[18] # percentile 95

        print(f"total sukses: {len(response_times)}")
        print(f"total error : {errors}")
        print(f"avg time    : {avg_time:.2f} ms")
        print(f"min time    : {min_time:.2f} ms")
        print(f"max time    : {max_time:.2f} ms")
        print(f"p95 time    : {p95_time:.2f} ms (95% request lebih cepat dari ini)")
        
        plt.figure(figsize=(12, 6))
        plt.plot(response_times, label='Response Time (ms)', color='blue', alpha=0.7)
        plt.axhline(avg_time, color='red', linestyle='--', label=f'Average ({avg_time:.2f} ms)')
        plt.axhline(p95_time, color='orange', linestyle=':', label=f'95th Percentile ({p95_time:.2f} ms)')
        plt.xlabel('Request Number')
        plt.ylabel('Response Time (ms)')
        plt.title('Server Response Time Across Requests')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('server_response_times.png')
        plt.show()
    else:
        print("semua request gagal")

if __name__ == "__main__":
    run_stress_test()