import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime

# ===================================================================
# FUNGSI-FUNGSI BANTU
# ===================================================================

def prediksi_cuaca(data_realtime, model, scaler_X, scaler_y):
    """Fungsi ini menjalankan prediksi cuaca menggunakan model yang telah dilatih."""
    features = ['TN', 'TX', 'RR', 'SS', 'FF_X']
    df_input = pd.DataFrame([data_realtime], columns=features)
    input_scaled = scaler_X.transform(df_input)
    pred_scaled = model.predict(input_scaled, verbose=0)
    pred_final = scaler_y.inverse_transform(pred_scaled)
    
    hasil_numerik = {
        'TAVG': pred_final[0][0],
        'RH_AVG': pred_final[0][1],
        'FF_AVG_KNOT': pred_final[0][2],
        'DDD_X': int(pred_final[0][3])
    }
    return hasil_numerik

def get_rekomendasi_sacha_inchi(prediksi_numerik, input_cuaca):
    """Fungsi ini memberikan rekomendasi penyiraman untuk tanaman Sacha Inchi."""
    skor = 0
    suhu, kelembapan, kecepatan_angin_knot, curah_hujan = (
        prediksi_numerik['TAVG'], prediksi_numerik['RH_AVG'],
        prediksi_numerik['FF_AVG_KNOT'], float(input_cuaca['RR'])
    )
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852

    if suhu > 30: skor += 3
    elif suhu >= 24: skor += 2
    else: skor += 1
    if kelembapan < 70: skor += 3
    elif kelembapan <= 85: skor += 2
    else: skor += 1
    if kecepatan_angin_kmh > 20: skor += 3
    elif kecepatan_angin_kmh >= 10: skor += 2
    else: skor += 1
    if curah_hujan > 5: skor -= 10
    elif curah_hujan >= 1: skor -= 4
    
    if skor <= 0: rekomendasi = "Tidak Perlu Penyiraman"
    elif skor <= 4: rekomendasi = "Penyiraman Ringan"
    elif skor <= 7: rekomendasi = "Penyiraman Sedang"
    else: rekomendasi = "Penyiraman Intensif"
    
    detail = f"Total Skor: {skor}"
    return rekomendasi, detail

def klasifikasi_cuaca(prediksi_numerik, input_cuaca):
    """Fungsi ini memberikan satu label klasifikasi cuaca."""
    suhu, kelembapan, kecepatan_angin_knot, curah_hujan = (
        prediksi_numerik['TAVG'], prediksi_numerik['RH_AVG'],
        prediksi_numerik['FF_AVG_KNOT'], float(input_cuaca['RR'])
    )
    if kecepatan_angin_knot < 0: kecepatan_angin_knot = 0
    kecepatan_angin_kmh = kecepatan_angin_knot * 1.852
    
    if curah_hujan > 10: klasifikasi = "Hujan Lebat"
    elif curah_hujan >= 1: klasifikasi = "Hujan Ringan"
    else:
        if kelembapan > 90: klasifikasi = "Berawan Tebal"
        elif kelembapan > 75: klasifikasi = "Cerah Berawan" if suhu <= 29 else "Panas dan Lembap"
        elif kelembapan > 60: klasifikasi = "Cerah" if suhu <= 30 else "Cerah dan Panas"
        else: klasifikasi = "Cerah dan Kering"

    if kecepatan_angin_kmh > 15:
        klasifikasi += " & Berangin"
    return klasifikasi

# === FUNGSI BARU UNTUK ARAH ANGIN ===
def klasifikasi_arah_angin(derajat):
    """Mengubah derajat arah angin menjadi teks (Utara, Tenggara, dll.)."""
    if 337.5 <= derajat <= 360 or 0 <= derajat < 22.5:
        return "Utara"
    elif 22.5 <= derajat < 67.5:
        return "Timur Laut"
    elif 67.5 <= derajat < 112.5:
        return "Timur"
    elif 112.5 <= derajat < 157.5:
        return "Tenggara"
    elif 157.5 <= derajat < 202.5:
        return "Selatan"
    elif 202.5 <= derajat < 247.5:
        return "Barat Daya"
    elif 247.5 <= derajat < 292.5:
        return "Barat"
    elif 292.5 <= derajat < 337.5:
        return "Barat Laut"
    else:
        return "Tidak Terdefinisi"

# ===================================================================
# BLOK EKSEKUSI UTAMA
# ===================================================================
try:
    # --- Langkah 1: Inisialisasi dan Muat Aset ---
    print("--- Memulai Proses ---")
    DATABASE_URL = 'https://tugas-akhir-64cd9-default-rtdb.asia-southeast1.firebasedatabase.app/' 
    cred = credentials.Certificate("tugas-akhir-64cd9-firebase-adminsdk-fbsvc-34e1efa674.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
    print("✅ Firebase Terhubung.")

    model = tf.keras.models.load_model('model_h20_p50.h5')
    scaler_X = joblib.load('scaler_X_4var.pkl')
    scaler_y = joblib.load('scaler_y_4var.pkl')
    print("✅ Model dan Scaler Dimuat.")

    # --- Langkah 2: Ambil & Proses Data Input ---
    ref_input = db.reference('/aws_01').order_by_key().limit_to_last(1)
    data_terbaru_dict = ref_input.get()

    if not data_terbaru_dict:
        print("❌ Tidak ada data sensor ditemukan.")
        exit()

    key = list(data_terbaru_dict.keys())[0]
    data_mentah = data_terbaru_dict[key]
    
    suhu_data = data_mentah.get('suhu', {})
    angin_data = data_mentah.get('angin', {})
    hujan_data = data_mentah.get('hujan', {})
    cahaya_data = data_mentah.get('cahaya', {})

    data_input_model = {
        'TN': float(suhu_data.get('min', 0.0)),
        'TX': float(suhu_data.get('max', 0.0)),
        'RR': float(hujan_data.get('total_harian_mm', 0.0)),
        'FF_X': float(angin_data.get('gust_kmh', 0.0)) * 0.54,
        'SS': float(cahaya_data.get('avg', 0.0))
    }

    # --- Langkah 3: Jalankan Semua Fungsi ---
    prediksi_numerik = prediksi_cuaca(data_input_model, model, scaler_X, scaler_y)
    rekomendasi, detail_skor = get_rekomendasi_sacha_inchi(prediksi_numerik, data_input_model)
    klasifikasi = klasifikasi_cuaca(prediksi_numerik, data_input_model)
    # Panggil fungsi baru untuk arah angin
    arah_angin_teks = klasifikasi_arah_angin(prediksi_numerik['DDD_X'])
    print("✅ Prediksi, Rekomendasi, dan Klasifikasi Selesai.")

    # --- Langkah 4: Tampilkan Hasil Akhir di Console ---
    kecepatan_angin_kmh_prediksi = prediksi_numerik['FF_AVG_KNOT'] * 1.852
    print("\n" + "="*40)
    print("--- HASIL PREDIKSI & REKOMENDASI ---")
    print(f"Klasifikasi Cuaca: {klasifikasi}")
    print(f"Prediksi Suhu: {prediksi_numerik['TAVG']:.2f} °C")
    print(f"Prediksi Kelembapan: {prediksi_numerik['RH_AVG']:.2f} %")
    print(f"Prediksi Kecepatan Angin: {kecepatan_angin_kmh_prediksi:.2f} km/jam")
    # Tampilkan arah angin dalam bentuk teks
    print(f"Prediksi Arah Angin: {arah_angin_teks} ({prediksi_numerik['DDD_X']}°)")
    print(f"Rekomendasi Penyiraman: {rekomendasi} ({detail_skor})")
    print("="*40)

    # --- Langkah 5: Simpan Hasil Gabungan ke Firebase ---
    timestamp_key = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    data_hasil_untuk_disimpan = {
        'prediksi_cuaca': {
            'suhu_avg_c': float(round(prediksi_numerik['TAVG'], 2)),
            'rh_avg_persen': float(round(prediksi_numerik['RH_AVG'], 2)),
            'ff_avg_kmh': float(round(kecepatan_angin_kmh_prediksi, 2)),
            'ddd_x_derajat': int(prediksi_numerik['DDD_X']),
            'ddd_x_teks': arah_angin_teks # Tambahkan arah angin teks
        },
        'rekomendasi_penyiraman': {
            'rekomendasi': rekomendasi,
            'detail_skor': detail_skor
        },
        'klasifikasi_cuaca': klasifikasi,
        'timestamp': timestamp_key
    }
    
    ref_output = db.reference(f'/Hasil_Prediksi_Rekomendasi_Penyiraman/{timestamp_key}')
    ref_output.set(data_hasil_untuk_disimpan)
    
    print(f"\n✅ Hasil gabungan berhasil disimpan ke Firebase di path '/Hasil_Prediksi_Rekomendasi_Penyiraman/{timestamp_key}'.")
    
except Exception as e:
    print(f"\n❌ Terjadi error pada proses utama: {e}")
