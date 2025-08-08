# Proyek Prediksi Cuaca dan Rekomendasi Penyiraman Otomatis

Proyek ini menggunakan model machine learning untuk memprediksi kondisi cuaca dan memberikan rekomendasi penyiraman. Proses ini dijalankan secara otomatis menggunakan GitHub Actions.

# Fitur
- **Prediksi Cuaca:** Memprediksi suhu, kelembapan, dan angin.
- **Rekomendasi Penyiraman:** Memberikan rekomendasi berdasarkan hasil prediksi.
- **Otomatisasi:** Berjalan secara otomatis sesuai jadwal via GitHub Actions.
- **Pencatatan Riwayat:** Menyimpan setiap hasil prediksi sebagai log di Firebase.

# Setup
1. **Clone Repositori:**
   ```bash
   git clone [https://github.com/fikrymaulana141/Tugas_Akhir_Prediksi_Cuaca_dan_Rekomendasi_Penyiraman.git](https://github.com/fikrymaulana141/Tugas_Akhir_Prediksi_Cuaca_dan_Rekomendasi_Penyiraman.git)
   ```
2. **Buat Secret:**
   - Buka repositori di GitHub, pergi ke **Settings > Secrets and variables > Actions**.
   - Buat secret baru bernama `FIREBASE_CREDENTIALS`.
   - Isinya adalah output dari perintah **Base64** pada file kredensial `.json` Anda.

# Status Workflow
![Prediksi Cuaca dan Rekomendasi Penyiraman](https://github.com/fikrymaulana141/Tugas_Akhir_Prediksi_Cuaca_dan_Rekomendasi_Penyiraman/actions/workflows/run_prediction.yml/badge.svg)
