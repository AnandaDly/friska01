# ===================================================================
# KODE APLIKASI STREAMLIT (PERBAIKAN FINAL UNTUK NameError)
# Nama File: app.py
# ===================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Sistem Prediksi Restock",
    page_icon="📦",
    layout="wide"
)

# --- Fungsi Prediksi (Tanpa Pelatihan) ---
def make_prediction(model, item_code_map, last_stock_map):
    """Fungsi untuk membuat prediksi menggunakan model yang sudah dilatih."""
    features = ['item_code', 'dayofweek', 'month', 'year', 'weekofyear']
    
    # Dapatkan tanggal hari ini untuk memulai prediksi
    start_date = pd.Timestamp.now()
    future_dates = pd.to_datetime([start_date + timedelta(days=i) for i in range(1, 8)])
    
    future_df_list = []
    
    # --- PERBAIKAN DIMULAI DI SINI: Menambahkan kembali perulangan tanggal yang hilang ---
    # Loop untuk setiap hari di masa depan
    for date in future_dates:
        # Loop untuk setiap barang yang dikenal oleh model
        for item_name, item_c in item_code_map.items():
            future_df_list.append({
                'TANGGAL': date, 
                'NAMA BARANG': item_name, 
                'item_code': item_c
            })
    # --- PERBAIKAN SELESAI ---
            
    future_df = pd.DataFrame(future_df_list)

    # Feature engineering untuk data masa depan
    future_df['dayofweek'] = future_df['TANGGAL'].dt.dayofweek
    future_df['month'] = future_df['TANGGAL'].dt.month
    future_df['year'] = future_df['TANGGAL'].dt.year
    future_week_series = future_df['TANGGAL'].dt.isocalendar().week
    future_df['weekofyear'] = future_week_series.astype(float).fillna(0).astype(int)
    
    # Prediksi
    X_future = future_df[features]
    future_predictions = model.predict(X_future)
    future_df['Prediksi_Jual'] = np.round(future_predictions).astype(int)
    future_df.loc[future_df['Prediksi_Jual'] < 0, 'Prediksi_Jual'] = 0

    # Agregasi hasil
    weekly_summary = future_df.groupby('NAMA BARANG')['Prediksi_Jual'].sum().reset_index()
    weekly_summary.rename(columns={'Prediksi_Jual': 'Prediksi_Jual_1_Minggu'}, inplace=True)
    
    # Membersihkan nama barang di hasil summary untuk memastikan kecocokan
    weekly_summary['NAMA BARANG'] = weekly_summary['NAMA BARANG'].str.strip()
    weekly_summary['Stok_Saat_Ini'] = weekly_summary['NAMA BARANG'].map(last_stock_map).fillna(0).astype(int)
    
    weekly_summary['Jumlah_Restock'] = weekly_summary['Prediksi_Jual_1_Minggu'] - weekly_summary['Stok_Saat_Ini']
    weekly_summary.loc[weekly_summary['Jumlah_Restock'] < 0, 'Jumlah_Restock'] = 0
    
    return weekly_summary.sort_values(by='Jumlah_Restock', ascending=False)

# --- Antarmuka Aplikasi Streamlit ---
st.title("📦 Sistem Prediksi Kebutuhan Restock Barang")
st.write("Aplikasi ini menggunakan model Machine Learning yang sudah dilatih untuk memberikan rekomendasi.")

# --- Memuat Model dan Mapping ---
try:
    model = joblib.load('model_prediksi.joblib')
    item_code_map = joblib.load('item_code_map.joblib')
    st.success("Model prediksi dan data item berhasil dimuat!")
except FileNotFoundError:
    st.error("File model (`model_prediksi.joblib` atau `item_code_map.joblib`) tidak ditemukan.")
    st.info("Harap pastikan file model berada di folder yang sama dengan aplikasi ini. Jalankan Notebook Training terlebih dahulu.")
    st.stop()

# --- Input dari Pengguna ---
st.header("Input Data Stok Terkini")
st.write("Unggah file CSV berisi penjualan/stok minggu lalu. Aplikasi ini akan menggunakan kolom `NAMA BARANG` dan `STOK AKHIR` untuk menghitung kebutuhan restock.")

uploaded_file = st.file_uploader(
    "Unggah data penjualan/stok terakhir (Format CSV)",
    type="csv"
)

if uploaded_file is not None:
    try:
        current_data_df = pd.read_csv(uploaded_file)
        
        required_columns = ['NAMA BARANG', 'STOK AKHIR']
        if not all(col in current_data_df.columns for col in required_columns):
            st.error(f"File CSV yang diunggah harus memiliki kolom: {', '.join(required_columns)}")
        else:
            st.success("File data stok berhasil diunggah.")
            
            # Membersihkan spasi ekstra dari nama barang di file yang diunggah
            current_data_df['NAMA BARANG'] = current_data_df['NAMA BARANG'].str.strip()

            # Logika baru yang lebih sederhana dan benar untuk membuat peta stok terakhir
            last_entries = current_data_df.groupby('NAMA BARANG').last()
            last_stock_map = last_entries['STOK AKHIR'].to_dict()
            
            # Tombol untuk memicu prediksi
            if st.button("Buat Prediksi Restock", type="primary"):
                with st.spinner("Membuat prediksi untuk 7 hari ke depan..."):
                    recommendation_df = make_prediction(model, item_code_map, last_stock_map)
                
                st.success("Rekomendasi berhasil dibuat!")
                st.subheader("Rekomendasi Restock untuk 1 Minggu ke Depan")
                st.dataframe(recommendation_df)
                
                # Opsi unduh
                csv_output = recommendation_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Unduh Rekomendasi sebagai CSV",
                    data=csv_output,
                    file_name='rekomendasi_restock.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    st.info("Silakan unggah file CSV data stok terkini untuk memulai.")