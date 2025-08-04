# Proyek Prediksi Cuaca dan Rekomendasi Penyiraman

Proyek ini menggunakan model machine learning untuk memprediksi kondisi cuaca dan memberikan rekomendasi penyiraman untuk lahan pertanian kacang Sacha Inchi. Skrip ini diotomatisasi menggunakan GitHub Actions untuk berjalan setiap 3 jam.

## Fitur
- **Prediksi Cuaca:** Memprediksi suhu, kelembapan, kecepatan, dan arah angin.
- **Klasifikasi Cuaca:** Memberikan label cuaca umum (e.g., "Cerah Berawan").
- **Rekomendasi Penyiraman:** Memberikan skor dan rekomendasi penyiraman berdasarkan hasil prediksi.
- **Otomatisasi:** Berjalan secara otomatis setiap 3 jam menggunakan GitHub Actions.
- **Penyimpanan Data:** Menyimpan hasil prediksi dan rekomendasi ke Firebase Realtime Database.

---

## Cara Kerja
1. **Pemicu (Trigger):** GitHub Actions akan menjalankan workflow secara otomatis setiap 3 jam.
2. **Setup Lingkungan:** Workflow akan menyiapkan lingkungan Python dan menginstall semua dependensi dari `requirements.txt`.
3. **Otentikasi:** Kredensial Firebase yang disimpan di GitHub Secrets akan digunakan untuk otentikasi.
4. **Eksekusi Skrip:** Skrip `Final_Prediksi_Penyiraman.py` akan dijalankan.
    - Mengambil data sensor terbaru dari Firebase.
    - Melakukan prediksi dan klasifikasi menggunakan model yang telah dilatih.
    - Menghasilkan rekomendasi penyiraman.
    - Menyimpan hasil kembali ke Firebase dengan *timestamp* sebagai kunci unik.

---

## Status Workflow
![Prediksi Cuaca dan Rekomendasi Penyiraman](https://github.com/FIKRY_MAULANA/Prediksi_Sacha_Inchi/actions/workflows/run_prediction.yml/badge.svg)
