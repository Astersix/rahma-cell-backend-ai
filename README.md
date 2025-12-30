# ⚙️ Backend Main Service - CV Rahma Cell

Repository ini adalah **Core Service** yang menangani seluruh logika bisnis, transaksi, dan manajemen data untuk Marketplace CV Rahma Cell. [cite_start]Layanan ini menghubungkan Frontend dengan Database dan Layanan AI\.

## 🛠 Tech Stack
* **Runtime:** Node.js & Express.js 
* **Database:** MySQL 
* **ORM:** Prisma 
* **Payment Gateway:** Midtrans (QRIS Integration) 

## 🔑 Fitur & Kapabilitas
* **RESTful API:** Menyediakan endpoint untuk manajemen produk, user, dan order.
* **Keamanan:** Autentikasi menggunakan JWT (JSON Web Token) dan proteksi password dengan Bcrypt
* **Role-Based Access Control (RBAC):** Pemisahan hak akses antara Admin dan Customer.
* **Integrasi AI:** Menghubungkan permintaan prediksi stok ke *AI Service*.

## 💻 Cara Menjalankan

1.  **Install dependencies:**
    ```bash
    npm install
    ```
2.  **Setup Database:**
    Pastikan MySQL sudah berjalan, lalu konfigurasi file `.env`:
    ```env
    DATABASE_URL="mysql://user:password@localhost:3306/rahmacell_db"
    JWT_SECRET="rahasia_anda"
    PORT=3000
    ```
3.  **Migrasi Database:**
    ```bash
    npx prisma migrate dev
    ```
4.  **Jalankan Server:**
    ```bash
    npm run start:dev
    ```

---
**Developed by Astersix Team**
