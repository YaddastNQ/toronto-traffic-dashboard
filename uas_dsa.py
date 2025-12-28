import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import folium
from streamlit_folium import st_folium

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Toronto Traffic",
    page_icon="🚦",
    layout="wide"
)

# --- 2. LOAD BUNDLE ---
@st.cache_resource
def load_model_bundle():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(curr_dir, 'toronto_flow_model_bundle.pkl')
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        return None

bundle = load_model_bundle()

# --- HEADER (TANPA FOTO) ---
st.title("🚦 Toronto Traffic")
st.markdown("Sistem pemantauan & prediksi lalu lintas Kota Toronto.")
st.divider()

if bundle is None:
    st.error("❌ File model tidak ditemukan.")
    st.stop()

# Bongkar Bundle
model = bundle["model"]
features_train = bundle["features"]
scaler = bundle.get("scaler")
global_mean = bundle["global_mean"]
detid_code_map = bundle["detid_code_map"]
detid_mean = bundle["detid_mean"]

# --- 3. GENERATE LOKASI (SIMULASI) ---
@st.cache_data
def generate_sensor_locations(code_map):
    np.random.seed(42) # Kunci koordinat
    sensors = []
    names = list(code_map.keys())
    sample_names = names[:50]
    
    for name in sample_names:
        lat = 43.65 + np.random.normal(0, 0.04)
        lon = -79.38 + np.random.normal(0, 0.06)
        code = code_map[name]
        sensors.append({"name": name, "code": code, "lat": lat, "lon": lon})
    
    return pd.DataFrame(sensors)

df_sensors = generate_sensor_locations(detid_code_map)

if 'selected_sensor_name' not in st.session_state:
    st.session_state['selected_sensor_name'] = df_sensors.iloc[0]['name']

# --- 4. LAYOUT UTAMA ---
col_left, col_right = st.columns([1, 1.5], gap="large")

# =========================================
# BAGIAN KIRI: PANEL KONTROL
# =========================================
with col_left:
    st.subheader("🎛️ Parameter Kondisi")
    
    with st.container(border=True):
        # 1. WAKTU
        hari_map = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            hari_input = st.selectbox("📅 Hari", list(hari_map.keys()))
        with col_t2:
            jam_val = st.slider("⏰ Jam", 0, 23, 8)
            
        dow_val = hari_map[hari_input]
        is_weekend = 1 if dow_val >= 5 else 0

        # 2. KEPADATAN (AUTO-PROFILE)
        def get_auto_occ(h, is_wk):
            if is_wk: return 0.40 if 11 <= h <= 18 else 0.05 if h >= 22 or h <= 6 else 0.20
            else: return 0.85 if 7 <= h <= 9 else 0.90 if 16 <= h <= 18 else 0.50 if 10 <= h <= 15 else 0.30
        
        auto_occ = get_auto_occ(jam_val, is_weekend)
        
        st.write("---")
        st.write("🚗 **Kepadatan Jalan**")
        
        mode_manual = st.checkbox("Ubah manual?", False)
        if mode_manual:
            occ_input = st.slider("Geser (0% - 100%)", 0, 100, int(auto_occ*100))
            occ_val = occ_input / 100.0
        else:
            st.progress(auto_occ)
            st.caption(f"Estimasi Kepadatan: **{int(auto_occ*100)}%** (Berdasarkan jam)")
            occ_val = auto_occ

        # TOMBOL
        st.write("")
        cek_btn = st.button("🚀 Prediksi", type="primary", use_container_width=True)
        
    st.info(f"📍 **Lokasi:** {st.session_state['selected_sensor_name']}")

    with st.expander("ℹ️ Info Status"):
        st.markdown("""
        - **🟢 LANCAR:** Aman, kecepatan normal.
        - **🟡 RAMAI:** Volume kendaraan tinggi, tetap waspada.
        - **🔴 MACET:** Hindari jika memungkinkan.
        """)

# =========================================
# BAGIAN KANAN: PETA & HASIL
# =========================================
with col_right:
    st.subheader("🗺️ Peta Pantauan")
    
    m = folium.Map(location=[43.65, -79.38], zoom_start=11, tiles="CartoDB positron")
    
    for _, row in df_sensors.iterrows():
        is_selected = row['name'] == st.session_state['selected_sensor_name']
        color = "red" if is_selected else "blue"
        icon_type = "car" if is_selected else "info-sign"
        
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"ID: {row['code']}",
            tooltip=row['name'], 
            icon=folium.Icon(color=color, icon=icon_type, prefix='fa')
        ).add_to(m)

    map_data = st_folium(m, height=350, use_container_width=True)

    if map_data.get("last_object_clicked_tooltip"):
        clicked_name = map_data["last_object_clicked_tooltip"]
        if clicked_name != st.session_state['selected_sensor_name']:
            st.session_state['selected_sensor_name'] = clicked_name
            st.rerun()

    # HASIL PREDIKSI
    if cek_btn:
        st.write("")
        st.subheader("📊 Hasil Prediksi")
        
        with st.spinner("Menghitung data..."):
            try:
                curr_name = st.session_state['selected_sensor_name']
                curr_code = df_sensors[df_sensors['name'] == curr_name]['code'].values[0]
                target_mean = detid_mean.get(curr_code, global_mean)

                # Feature Engineering
                time_sin = np.sin(2 * np.pi * jam_val / 24.0)
                time_cos = np.cos(2 * np.pi * jam_val / 24.0)
                dow_sin = np.sin(2 * np.pi * dow_val / 7.0)
                dow_cos = np.cos(2 * np.pi * dow_val / 7.0)
                
                raw_data = {
                    'detid_code': [curr_code],
                    'occ': [occ_val], 'hour': [jam_val],
                    'day_of_week': [dow_val], 'is_weekend': [is_weekend],
                    'detid_mean_flow': [target_mean], 'detid_mean_target': [target_mean],
                    'time_sin': [time_sin], 'time_cos': [time_cos],
                    'dow_sin': [dow_sin], 'dow_cos': [dow_cos],
                    'flow_lag1': [target_mean], 'flow_lag2': [target_mean],
                    'flow_roll4': [target_mean], 'occ_lag1': [occ_val]
                }
                
                df_input = pd.DataFrame(raw_data)
                df_ready = pd.DataFrame()
                for f in features_train:
                    df_ready[f] = df_input[f] if f in df_input.columns else 0.0
                
                if scaler: X_final = scaler.transform(df_ready)
                else: X_final = df_ready
                
                pred = int(max(0, model.predict(X_final)[0]))
                
                # --- LOGIKA 3 STATUS ---
                if occ_val > 0.8:
                    status = "MACET 🔴"
                    tips = "⛔ **Saran:** Hindari jalan ini. Terjadi penumpukan kendaraan parah. Cari rute alternatif atau naik transportasi umum."
                elif pred < 300:
                    status = "LANCAR 🟢"
                    tips = "✅ **Saran:** Jalanan aman terkendali. Waktu yang tepat untuk melintas."
                else:
                    status = "RAMAI 🟡"
                    tips = "⚠️ **Saran:** Volume kendaraan tinggi tapi tetap jalan. Tetap fokus dan jaga jarak aman."
                
                # Output
                with st.container(border=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Estimasi Volume", f"{pred}", "Unit/Jam")
                    with c2:
                        st.metric("Status Jalan", status)
                    
                    st.divider()
                    st.markdown(tips)
                
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("© 2025 Kelompok Toronto | Data Science & Analysis | UAS Semester 7 | Telkom University")