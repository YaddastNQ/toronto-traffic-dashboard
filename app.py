import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Toronto Traffic Monitor", page_icon="🚦", layout="wide")

# --- 2. LOAD DATA ---
@st.cache_resource
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'traffic_rf_model.pkl')
    scaler_path = os.path.join(current_dir, 'traffic_scaler.pkl')
    if not os.path.exists(model_path): st.stop()
    with open(model_path, 'rb') as f: model = joblib.load(f)
    with open(scaler_path, 'rb') as f: scaler = joblib.load(f)
    return model, scaler

model, scaler = load_data()

# --- 3. HEADER (LOGO + TITLE) ---
c_logo, c_title = st.columns([1, 10])
with c_logo:
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logo.png')
    if os.path.exists(logo_path): st.image(logo_path, use_container_width=True)
with c_title:
    st.title("🚦Toronto Traffic Monitor")
    st.markdown("Simulasi prediksi kemacetan di Toronto")

st.write("") # Spacer

# --- 4. INPUT AREA (HORIZONTAL STYLE) ---
# Kita bungkus dalam container biar rapi
with st.container(border=True):
    st.subheader("🛠️ Parameter")
    
    col_in1, col_in2, col_in3, col_btn = st.columns([2, 2, 2, 1])
    
    with col_in1:
        occ = st.slider("📉 Kepadatan (Occupancy)", 0.0, 0.5, 0.02, 0.001)
    with col_in2:
        hour = st.slider("⏰ Jam Operasional", 0, 23, 8)
    with col_in3:
        hari_dict = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
        hari_label = st.selectbox("📅 Hari", list(hari_dict.keys()))
        day_of_week = hari_dict[hari_label]
        is_weekend = 1 if day_of_week >= 5 else 0
    with col_btn:
        st.write("") # Spacer biar tombol sejajar
        st.write("") 
        predict_btn = st.button("PREDIKSI", type="primary", use_container_width=True)

# --- 5. RESULT AREA ---
if predict_btn:
    st.divider()
    
    # Logic Prediksi
    input_data = scaler.transform(np.array([[occ, hour, day_of_week, is_weekend]]))
    prediction = model.predict(input_data)[0]

    # --- LOGIC STATUS YANG LEBIH DETAIL ---
    if prediction < 100:
        status = "LANCAR JAYA"
        color = "green"
        icon = "🟢"
        # Analisis Situasi
        pesan = "Kondisi lalu lintas sangat ideal. Kapasitas jalan masih lancar dan kecepatan rata-rata kendaraan diprediksi tinggi."
        # Rekomendasi Aksi
        saran = "✅ **Rekomendasi:** Waktu yang tepat untuk melakukan perjalanan logistik atau bepergian tanpa hambatan."
    
    elif prediction < 250:
        status = "RAMAI LANCAR"
        color = "orange"
        icon = "🟡"
        # Analisis Situasi
        pesan = "Volume kendaraan meningkat signifikan mendekati jam sibuk. Arus masih mengalir, namun terjadi perlambatan kecepatan."
        # Rekomendasi Aksi
        saran = "⚠️ **Rekomendasi:** Tetap waspada dan jaga jarak aman. Estimasi keterlambatan ringan (10-15 menit) mungkin terjadi."
    
    else:
        status = "MACET PARAH"
        color = "red"
        icon = "🔴"
        # Analisis Situasi
        pesan = "Terdeteksi penumpukan kendaraan ekstrem. Rasio volume terhadap kapasitas jalan sudah jenuh."
        # Rekomendasi Aksi
        saran = "🛑 **Rekomendasi:** Sangat disarankan mencari rute alternatif atau menunda perjalanan hingga volume menurun."

    # --- TAMPILAN HASIL (Update bagian ini juga di bawahnya) ---
    c_res1, c_res2 = st.columns([1, 1])
    
    with c_res1:
        st.markdown(f"### Status: :{color}[{status}]")
        st.metric("Estimasi Volume", f"{int(prediction)} Unit", f"{icon} {status}")
        
        # Visualisasi Progress Bar dengan warna sesuai kondisi
        st.progress(min(prediction/400, 1.0))
        st.caption(f"Load Jalan: {min(prediction/400, 1.0)*100:.0f}%")
        
    with c_res2:
        # Menampilkan Pesan Detail tadi
        if color == "green":
            st.success(f"{pesan}")
            st.info(saran)
        elif color == "orange":
            st.warning(f"{pesan}")
            st.info(saran)
        else:
            st.error(f"{pesan}")
            st.error(saran) # Pakai error box juga biar merah tebal

        with st.expander("Lihat Data Mentah"):
            st.dataframe(pd.DataFrame({
                "Occupancy": [occ], "Jam": [hour], "Hari": [hari_label], "Prediksi": [prediction]
            }), hide_index=True)

else:
    st.info("👆 Masukkan parameter di atas lalu tekan tombol Prediksi.")

# --- FOOTER ---
st.markdown("---") # Garis pemisah

st.caption("© 2025 Telkom University | Developed by Kelompok Toronto | TK-46-03 | Data Science & Analysis ")
