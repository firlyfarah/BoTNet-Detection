import streamlit as st
import numpy as np
import joblib
import gdown
import os

from modelsvm import SVM  

st.set_page_config(page_title="IoTect", layout="wide")

model_path = "SVM_MODEL.pkl"
gdrive_file_id = "1pfFw4AA-zqakXGHFphUeH-DxExBz891Y"
if not os.path.exists(model_path):
    with st.spinner("🔽 Mengunduh model..."):
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, model_path, quiet=False)

st.markdown("""
    <style>
    .header-container {
        background-color: #0B2C54;
        padding: 40px 0 30px 0;
        text-align: center;
        color: white;
        width: 100%;
        margin-top: -3rem;
    }
    </style>
    <div class="header-container">
        <h1>IoTect</h1>
        <h3>BotNet Attack Detection on IoT Networks</h3>
        <p>Masukkan parameter lalu lintas jaringan IoT Anda untuk mendeteksi potensi serangan BotNet secara otomatis</p>
    </div>
""", unsafe_allow_html=True)

with st.expander("📘 Keterangan Singkatan Fitur"):
    st.markdown("""
    - **MI**: *Mutual Information*  
    - **H**: *Host*  
    - **HH**: *Host-to-Host Communication*  
    - **HH_jit**: *Jitter antar Host*  
    - **HpHp**: *Host:Port to Host:Port Communication*  
    """)

st.markdown("---")

feature_names = [
    "MI_dir_L0.1_weight", "MI_dir_L0.1_mean", "MI_dir_L0.1_variance",
    "H_L0.1_weight", "H_L0.1_mean", "H_L0.1_variance",
    "HH_L0.1_weight", "HH_L0.1_mean", "HH_L0.1_std",
    "HH_L0.1_magnitude", "HH_L0.1_radius", "HH_L0.1_covariance", "HH_L0.1_pcc",
    "HH_jit_L0.1_weight", "HH_jit_L0.1_mean", "HH_jit_L0.1_variance",
    "HpHp_L0.1_weight", "HpHp_L0.1_mean", "HpHp_L0.1_std",
    "HpHp_L0.1_magnitude", "HpHp_L0.1_radius", "HpHp_L0.1_covariance", "HpHp_L0.1_pcc"
]

col1, col2, col3 = st.columns(3)
user_inputs = []

for i in range(6):
    val = col1.text_input(feature_names[i])
    user_inputs.append(val)
for i in range(6, 12):
    val = col2.text_input(feature_names[i])
    user_inputs.append(val)
for i in range(12, 18):
    val = col3.text_input(feature_names[i])
    user_inputs.append(val)

if st.button("PREDICT"):
    try:
        if "" in user_inputs:
            st.warning("⚠️ Semua input harus diisi.")
        else:
            input_data = np.array([float(x) for x in user_inputs]).reshape(1, -1)
            model = joblib.load(model_path)
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.error("🚨 BotNet Attack Detected!")
            else:
                st.success("✅ Normal Traffic - No Attack Detected.")
    except ValueError as e:
        st.warning(f"⚠️ Pastikan semua input diisi dengan angka. Error: {e}")
    except FileNotFoundError:
        st.error("❌ Model file tidak ditemukan.")
