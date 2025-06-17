import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="🛒 Sistem Prediksi Restock Barang",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk style penjualan
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(45deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown("""
<div class="main-header">
    <h1>🛒 Sistem Prediksi Restock Barang</h1>
    <p>📊 Analisis Penjualan & Prediksi Kebutuhan Stok Minggu Depan</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Menu Navigasi")
    menu = st.selectbox(
        "Pilih Menu:",
        ["🏠 Dashboard", "📤 Upload Data", "🔮 Prediksi", "📊 Analisis", "💾 Download Hasil"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ Informasi Sistem")
    st.info("""
    **Format Data yang Diperlukan:**
    - MINGGU
    - TANGGAL (format: DD/MM/YYYY)
    - KATEGORI
    - NAMA BARANG
    - SATUAN
    - STOK AWAL
    - JUMLAH TERJUAL
    - STOK AKHIR
    """)
    
    st.markdown("### 🎯 Fitur Utama")
    st.markdown("""
    ✅ Upload file Excel/CSV  
    ✅ Feature Engineering otomatis  
    ✅ Prediksi ML dengan RandomForest  
    ✅ Visualisasi interaktif  
    ✅ Export hasil prediksi  
    """)

# Fungsi utility
def load_model():
    """Load model ML yang sudah ditraining"""
    try:
        # Ganti dengan path model Anda
        model_data = joblib.load('model_restock.joblib')
        
        # Jika model_data adalah dictionary, ambil model dari key tertentu
        if isinstance(model_data, dict):
            # Coba beberapa key yang umum digunakan
            possible_keys = ['model', 'rf_model', 'regressor', 'estimator', 'best_estimator_']
            for key in possible_keys:
                if key in model_data:
                    return model_data[key]
            # Jika tidak ada key yang cocok, tampilkan keys yang tersedia
            st.error(f"⚠️ Model dalam format dictionary. Keys tersedia: {list(model_data.keys())}")
            st.info("💡 Silakan sesuaikan key di fungsi load_model() sesuai dengan struktur model Anda")
            return None
        else:
            # Jika bukan dictionary, return langsung
            return model_data
            
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None

def validate_data(df):
    """Validasi data input"""
    required_columns = [
        'MINGGU', 'TANGGAL', 'KATEGORI', 'NAMA BARANG', 
        'SATUAN', 'STOK AWAL', 'JUMLAH TERJUAL', 'STOK AKHIR'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Kolom yang hilang: {', '.join(missing_columns)}"
    
    # Validasi tipe data numerik
    numeric_columns = ['MINGGU', 'STOK AWAL', 'JUMLAH TERJUAL', 'STOK AKHIR']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return True, "Data valid"

def feature_engineering(df):
    """Melakukan feature engineering sesuai dengan model"""
    try:
        df_processed = df.copy()
        
        # Convert tanggal
        df_processed['TANGGAL'] = pd.to_datetime(df_processed['TANGGAL'], format='%d/%m/%Y', errors='coerce')
        
        # Extract features dari tanggal
        df_processed['TAHUN'] = df_processed['TANGGAL'].dt.year
        df_processed['BULAN'] = df_processed['TANGGAL'].dt.month
        df_processed['HARI_DALAM_BULAN'] = df_processed['TANGGAL'].dt.day
        
        # Sort data berdasarkan nama barang dan tanggal
        df_processed = df_processed.sort_values(['NAMA BARANG', 'TANGGAL']).reset_index(drop=True)
        
        # Feature engineering untuk data historis
        df_processed['PREV_JUMLAH_TERJUAL'] = df_processed.groupby('NAMA BARANG')['JUMLAH TERJUAL'].shift(1)
        df_processed['PREV_STOK_AKHIR'] = df_processed.groupby('NAMA BARANG')['STOK AKHIR'].shift(1)
        
        # Future sales untuk training (jika ada data masa depan)
        df_processed['FUTURE_SALES'] = df_processed.groupby('NAMA BARANG')['JUMLAH TERJUAL'].shift(-1)
        df_processed['RESTOCK_NEEDED'] = df_processed['FUTURE_SALES'] - df_processed['STOK AKHIR']
        df_processed['RESTOCK_NEEDED'] = df_processed['RESTOCK_NEEDED'].clip(lower=0)
        
        # Encoding kategori dan nama barang
        le_kategori = LabelEncoder()
        le_nama_barang = LabelEncoder()
        
        df_processed['KATEGORI_ENCODED'] = le_kategori.fit_transform(df_processed['KATEGORI'])
        df_processed['NAMA_BARANG_ENCODED'] = le_nama_barang.fit_transform(df_processed['NAMA BARANG'])
        
        # Simpan decoder untuk output
        df_processed['KATEGORI_DECODE'] = df_processed['KATEGORI']
        df_processed['NAMA_BARANG_DECODE'] = df_processed['NAMA BARANG']
        
        return df_processed, le_kategori, le_nama_barang, True, "Feature engineering berhasil"
        
    except Exception as e:
        return None, None, None, False, f"Error dalam feature engineering: {str(e)}"

def predict_restock(df_processed, model):
    """Melakukan prediksi restock"""
    try:
        # Features yang digunakan untuk prediksi
        features = [
            'MINGGU', 'KATEGORI_ENCODED', 'NAMA_BARANG_ENCODED',
            'STOK AWAL', 'JUMLAH TERJUAL', 'STOK AKHIR', 'BULAN', 'TAHUN', 'HARI_DALAM_BULAN'
        ]
        
        # Ambil data untuk prediksi (data terbaru per item)
        latest_data = df_processed.groupby('NAMA BARANG').last().reset_index()
        
        # Prediksi
        X_pred = latest_data[features]
        predictions = model.predict(X_pred)
        
        # Bulatkan ke angka bulat
        predictions = np.round(predictions).astype(int)
        predictions = np.maximum(predictions, 0)  # Pastikan tidak negatif
        
        # Tambahkan kolom prediksi
        latest_data['REKOMENDASI_RESTOCK'] = predictions
        
        return latest_data, True, "Prediksi berhasil"
        
    except Exception as e:
        return None, False, f"Error dalam prediksi: {str(e)}"

# Inisialisasi session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Menu Dashboard
if menu == "🏠 Dashboard":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <h3>📊 Status Sistem</h3>
            <p>Siap untuk prediksi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <h3>🤖 Model ML</h3>
            <p>RandomForest Regressor</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <h3>⏱️ Prediksi</h3>
            <p>1 Minggu ke Depan</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Cara Penggunaan")
    st.markdown("""
    1. **📤 Upload Data**: Upload file Excel/CSV dengan data penjualan
    2. **🔮 Prediksi**: Sistem akan melakukan feature engineering dan prediksi otomatis
    3. **📊 Analisis**: Lihat visualisasi dan analisis hasil prediksi
    4. **💾 Download**: Download hasil prediksi dalam format Excel/CSV
    """)
    
    # Load model status
    model = load_model()
    if model:
        st.success("✅ Model ML berhasil dimuat dan siap digunakan!")
    else:
        st.error("❌ Model ML tidak dapat dimuat. Silakan periksa file model.")

# Menu Upload Data
elif menu == "📤 Upload Data":
    st.markdown("### 📁 Upload File Data Penjualan")
    
    uploaded_file = st.file_uploader(
        "Pilih file Excel atau CSV",
        type=['xlsx', 'xls', 'csv'],
        help="Upload file dengan data penjualan yang berisi kolom: MINGGU, TANGGAL, KATEGORI, NAMA BARANG, SATUAN, STOK AWAL, JUMLAH TERJUAL, STOK AKHIR"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Validasi data
            is_valid, message = validate_data(df)
            
            if is_valid:
                st.session_state.uploaded_data = df
                
                st.markdown("""
                <div class="success-box">
                    <h4>✅ Data berhasil diupload!</h4>
                    <p>File telah divalidasi dan siap untuk diproses.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preview data
                st.markdown("### 👀 Preview Data")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Statistik data
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Baris", len(df))
                with col2:
                    st.metric("🛍️ Jumlah Barang", df['NAMA BARANG'].nunique())
                with col3:
                    st.metric("📂 Jumlah Kategori", df['KATEGORI'].nunique())
                with col4:
                    st.metric("📅 Periode Data", df['MINGGU'].nunique())
                
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>⚠️ Data tidak valid!</h4>
                    <p>{message}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ Error membaca file: {str(e)}")

# Menu Prediksi
elif menu == "🔮 Prediksi":
    st.markdown("### 🤖 Prediksi Restock Barang")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu Upload Data")
    else:
        model = load_model()
        if model is None:
            st.error("❌ Model tidak dapat dimuat. Silakan periksa file model.")
        else:
            if st.button("🎯 Mulai Prediksi", type="primary"):
                with st.spinner("🔄 Sedang memproses data dan melakukan prediksi..."):
                    # Feature engineering
                    df_processed, le_kat, le_nama, success, message = feature_engineering(st.session_state.uploaded_data)
                    
                    if success:
                        st.session_state.processed_data = df_processed
                        
                        # Prediksi
                        results, pred_success, pred_message = predict_restock(df_processed, model)
                        
                        if pred_success:
                            st.session_state.prediction_results = results
                            
                            st.markdown("""
                            <div class="success-box">
                                <h4>🎉 Prediksi Berhasil!</h4>
                                <p>Sistem telah menyelesaikan prediksi restock untuk semua barang.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Tampilkan hasil prediksi
                            st.markdown("### 📋 Hasil Prediksi Restock")
                            
                            # Filter untuk tampilan
                            display_columns = [
                                'KATEGORI', 'NAMA BARANG', 'SATUAN', 'STOK AKHIR', 
                                'JUMLAH TERJUAL', 'REKOMENDASI_RESTOCK'
                            ]
                            
                            results_display = results[display_columns].copy()
                            st.dataframe(results_display, use_container_width=True)
                            
                            # Statistik prediksi
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📦 Total Barang", len(results))
                            with col2:
                                st.metric("📈 Avg Restock", f"{results['REKOMENDASI_RESTOCK'].mean():.0f}")
                            with col3:
                                st.metric("🔝 Max Restock", results['REKOMENDASI_RESTOCK'].max())
                            with col4:
                                total_restock = results['REKOMENDASI_RESTOCK'].sum()
                                st.metric("🎯 Total Restock", total_restock)
                        
                        else:
                            st.error(f"❌ {pred_message}")
                    else:
                        st.error(f"❌ {message}")

# Menu Analisis
elif menu == "📊 Analisis":
    st.markdown("### 📈 Analisis Hasil Prediksi")
    
    if st.session_state.prediction_results is None:
        st.warning("⚠️ Belum ada hasil prediksi. Silakan lakukan prediksi terlebih dahulu.")
    else:
        results = st.session_state.prediction_results
        
        # Chart 1: Rekomendasi Restock per Kategori
        st.markdown("#### 📊 Rekomendasi Restock per Kategori")
        kategori_summary = results.groupby('KATEGORI')['REKOMENDASI_RESTOCK'].sum().reset_index()
        
        fig1 = px.bar(
            kategori_summary, 
            x='KATEGORI', 
            y='REKOMENDASI_RESTOCK',
            title="Total Rekomendasi Restock per Kategori",
            color='REKOMENDASI_RESTOCK',
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Chart 2: Top 10 Barang dengan Restock Tertinggi
        st.markdown("#### 🔝 Top 10 Barang dengan Rekomendasi Restock Tertinggi")
        top_items = results.nlargest(10, 'REKOMENDASI_RESTOCK')
        
        fig2 = px.bar(
            top_items,
            x='REKOMENDASI_RESTOCK',
            y='NAMA BARANG',
            orientation='h',
            title="Top 10 Barang - Rekomendasi Restock",
            color='REKOMENDASI_RESTOCK',
            color_continuous_scale='Reds'
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Chart 3: Scatter Plot Stok vs Rekomendasi
        st.markdown("#### 🎯 Hubungan Stok Akhir vs Rekomendasi Restock")
        fig3 = px.scatter(
            results,
            x='STOK AKHIR',
            y='REKOMENDASI_RESTOCK',
            color='KATEGORI',
            hover_data=['NAMA BARANG', 'JUMLAH TERJUAL'],
            title="Stok Akhir vs Rekomendasi Restock"
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Insight
        st.markdown("#### 💡 Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Kategori dengan Restock Tertinggi:**")
            top_kategori = kategori_summary.loc[kategori_summary['REKOMENDASI_RESTOCK'].idxmax(), 'KATEGORI']
            st.success(f"🏆 {top_kategori}")
            
            st.markdown("**⚠️ Barang Kritis (Restock > 50):**")
            critical_items = len(results[results['REKOMENDASI_RESTOCK'] > 50])
            st.warning(f"📦 {critical_items} barang")
        
        with col2:
            st.markdown("**💰 Estimasi Total Investasi:**")
            # Asumsi harga rata-rata per unit (bisa disesuaikan)
            avg_price = 10000  # 10rb per unit
            total_investment = results['REKOMENDASI_RESTOCK'].sum() * avg_price
            st.info(f"💵 Rp {total_investment:,.0f}")
            
            st.markdown("**📊 Efisiensi Stok:**")
            efficient_items = len(results[results['REKOMENDASI_RESTOCK'] <= 10])
            st.success(f"✅ {efficient_items} barang efisien")

# Menu Download
elif menu == "💾 Download Hasil":
    st.markdown("### 📥 Download Hasil Prediksi")
    
    if st.session_state.prediction_results is None:
        st.warning("⚠️ Belum ada hasil prediksi untuk didownload.")
    else:
        results = st.session_state.prediction_results
        
        st.markdown("### 📊 Preview Hasil Download")
        st.dataframe(results, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download Excel
            from io import BytesIO
            
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results.to_excel(writer, sheet_name='Prediksi_Restock', index=False)
            
            excel_data = output.getvalue()
            
            st.download_button(
                label="📊 Download Excel (.xlsx)",
                data=excel_data,
                file_name=f"prediksi_restock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Download CSV
            csv_data = results.to_csv(index=False)
            
            st.download_button(
                label="📄 Download CSV",
                data=csv_data,
                file_name=f"prediksi_restock_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ File Siap Download!</h4>
            <p>Hasil prediksi telah disiapkan dalam format Excel dan CSV. Klik tombol di atas untuk mendownload.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Statistik final
        st.markdown("### 📈 Ringkasan Hasil")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📦 Total Barang", len(results))
        with col2:
            st.metric("🔢 Total Restock", results['REKOMENDASI_RESTOCK'].sum())
        with col3:
            st.metric("📂 Kategori", results['KATEGORI'].nunique())
        with col4:
            avg_restock = results['REKOMENDASI_RESTOCK'].mean()
            st.metric("📊 Rata-rata Restock", f"{avg_restock:.1f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white; margin-top: 2rem;">
    <h4>🛒 Sistem Prediksi Restock Barang</h4>
    <p>Powered by Machine Learning • RandomForest Algorithm • Streamlit Framework</p>
    <p>📧 Untuk support teknis, hubungi tim IT Anda</p>
</div>
""", unsafe_allow_html=True)