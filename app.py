import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Prediksi Diabetes Menggunakan Machine Learning")
st.write(
    "Aplikasi ini digunakan untuk memprediksi kemungkinan diabetes "
    "berdasarkan data kesehatan pasien menggunakan beberapa model "
    "machine learning."
)

st.write("---")

@st.cache_resource
def load_pickle(file_path, file_label):
    """
    Memuat file .pkl menggunakan joblib.
    - file_path  : lokasi file
    - file_label : nama label untuk pesan error
    """
    try:
        if not os.path.exists(file_path):
            st.error(
                f"File {file_label} tidak ditemukan di:\n`{file_path}`\n\n"
                "Pastikan file tersedia dan folder sudah benar."
            )
            st.stop()
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Gagal memuat {file_label}: {e}")
        st.stop()


def get_model_path(model_name):
    """
    Mengembalikan path file model (.pkl) berdasarkan nama model
    yang dipilih user pada selectbox.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_paths = {
        "Decision Tree": os.path.join(base_dir, "Model", "DecisionTree", "best_dcs_model.pkl"),
        "KNN":           os.path.join(base_dir, "Model", "KNN",           "best_model_knn.pkl"),
        "Neural Network":os.path.join(base_dir, "Model", "NN",            "best_nn_model.pkl"),
        "SVM":           os.path.join(base_dir, "Model", "SVM",           "best_svm_model.pkl"),
    }
    return model_paths.get(model_name)


# ============================================================
# Deskripsi Singkat Tiap Model
# ============================================================
MODEL_DESCRIPTIONS = {
    "Decision Tree": (
        "Pohon keputusan yang memecah data berdasarkan fitur paling "
        "informatif. Mudah dipahami, namun rentan overfitting."
    ),
    "KNN": (
        "Klasifikasi berdasarkan tetangga terdekat. Sederhana dan efektif, "
        "namun lambat untuk dataset besar."
    ),
    "Neural Network": (
        "Jaringan saraf tiruan berlapis yang mampu menangkap pola "
        "non-linear kompleks. Akurat, tetapi kurang transparan."
    ),
    "SVM": (
        "Mencari hyperplane terbaik yang memisahkan kelas. Sangat "
        "efektif untuk data berdimensi tinggi."
    ),
}

# Interpretasi hasil prediksi (0 = Tidak Diabetes, 1 = Diabetes)
PREDICTION_INTERPRETATION = {
    0: {
        "emoji": "✅",
        "title": "Tidak Diabetes",
        "description": (
            "Berdasarkan data yang dimasukkan, model memprediksi bahwa "
            "pasien **kemungkinan besar tidak menderita diabetes**."
        ),
        "color": "success"
    },
    1: {
        "emoji": "⚠️",
        "title": "Diabetes",
        "description": (
            "Berdasarkan data yang dimasukkan, model memprediksi bahwa "
            "pasien **kemungkinan menderita diabetes**. Disarankan untuk "
            "melakukan konsultasi lebih lanjut dengan tenaga medis."
        ),
        "color": "error"
    },
}

st.sidebar.title("⚙️ Pengaturan Aplikasi")
st.sidebar.subheader("Prediksi Diabetes ML")

# Pilihan model di sidebar
model_option = st.sidebar.selectbox(
    "Pilih Model Machine Learning:",
    ("Decision Tree", "KNN", "Neural Network", "SVM")
)

# Informasi fitur input di sidebar
st.sidebar.write("### 📋 Fitur Input")
st.sidebar.info(
    """
    1. Pregnancies
    2. Glucose
    3. BloodPressure
    4. SkinThickness
    5. Insulin
    6. BMI
    7. DiabetesPedigreeFunction
    8. Age
    """
)

# Keterangan label
st.sidebar.write("### 🏷️ Keterangan Label")
st.sidebar.write("• **0** = Tidak Diabetes")
st.sidebar.write("• **1** = Diabetes")

st.sidebar.write("---")
st.sidebar.caption("Dibuat untuk Tugas Besar Machine Learning.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

imputer_path = os.path.join(BASE_DIR, "Imputer", "imputer_diabetes.pkl")
scaler_path  = os.path.join(BASE_DIR, "Scaler",  "scaler_diabetes.pkl")
model_path   = get_model_path(model_option)

# Load imputer dan scaler
imputer = load_pickle(imputer_path, "Imputer")
scaler  = load_pickle(scaler_path,  "Scaler")

# Load model sesuai pilihan user
model_name = os.path.splitext(os.path.basename(model_path))[0]
model      = load_pickle(model_path, f"Model {model_option} ({model_name})")

# Tab mode prediksi
tab1, tab2 = st.tabs(["📝 Prediksi Tunggal", "📂 Prediksi Batch (CSV)"])

# ============================================================
# TAB 1: PREDIKSI TUNGGAL
# ============================================================
with tab1:
    st.subheader("📝 Masukkan Data Pasien")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            pregnancies = st.number_input(
                "Pregnancies (Jumlah Kehamilan)",
                min_value=0, max_value=20, value=0, step=1
            )
            glucose = st.number_input(
                "Glucose (Kadar Glukosa)",
                min_value=0, max_value=300, value=120, step=1
            )
            blood_pressure = st.number_input(
                "BloodPressure (Tekanan Darah)",
                min_value=0, max_value=200, value=70, step=1
            )
            skin_thickness = st.number_input(
                "SkinThickness (Ketebalan Kulit)",
                min_value=0, max_value=100, value=20, step=1
            )

        with col2:
            insulin = st.number_input(
                "Insulin (Kadar Insulin)",
                min_value=0, max_value=900, value=80, step=1
            )
            bmi = st.number_input(
                "BMI (Indeks Massa Tubuh)",
                min_value=0.0, max_value=70.0, value=25.0,
                step=0.1, format="%.1f"
            )
            dpf = st.number_input(
                "DiabetesPedigreeFunction",
                min_value=0.0, max_value=3.0, value=0.5,
                step=0.01, format="%.2f"
            )
            age = st.number_input(
                "Age (Usia)",
                min_value=0, max_value=120, value=30, step=1
            )

        # Tombol prediksi di dalam form
        submit = st.form_submit_button("🔍 Prediksi")

    if submit:
        invalid_fields = []
        if glucose == 0:
            invalid_fields.append("Glucose")
        if blood_pressure == 0:
            invalid_fields.append("BloodPressure")
        if bmi == 0:
            invalid_fields.append("BMI")
        if age == 0:
            invalid_fields.append("Age")

        if invalid_fields:
            st.warning(
                f"⚠️ Nilai 0 terdeteksi pada kolom: **{', '.join(invalid_fields)}**.\n\n"
                "Mohon masukkan nilai yang valid (tidak boleh 0) untuk kolom tersebut."
            )
        else:

            data = {
                "Pregnancies":            [pregnancies],
                "Glucose":                [glucose],
                "BloodPressure":          [blood_pressure],
                "SkinThickness":          [skin_thickness],
                "Insulin":                [insulin],
                "BMI":                    [bmi],
                "DiabetesPedigreeFunction":[dpf],
                "Age":                    [age],
            }
            input_df = pd.DataFrame(data)

            cols_zero_to_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            input_df[cols_zero_to_nan] = input_df[cols_zero_to_nan].replace(0, np.nan)

            try:
                input_imputed = imputer.transform(input_df)
            except Exception as e:
                st.error(f"Gagal saat proses imputasi: {e}")
                st.stop()

            try:
                input_scaled = scaler.transform(input_imputed)
            except Exception as e:
                st.error(f"Gagal saat proses scaling: {e}")
                st.stop()

            try:
                prediction = model.predict(input_scaled)
            except Exception as e:
                st.error(f"Gagal saat melakukan prediksi: {e}")
                st.stop()

            st.write("---")
            st.subheader("📊 Hasil Prediksi")

            # Ambil hasil prediksi dan interpretasi
            result_key = int(prediction[0])
            interpretation = PREDICTION_INTERPRETATION[result_key]

            # Tampilkan hasil prediksi dengan warna yang sesuai
            if interpretation["color"] == "success":
                st.success(
                    f"{interpretation['emoji']} Hasil Prediksi: "
                    f"**{interpretation['title']}**"
                )
            else:
                st.error(
                    f"{interpretation['emoji']} Hasil Prediksi: "
                    f"**{interpretation['title']}**"
                )

            # Interpretasi hasil dalam bahasa sederhana
            st.write(f"📝 **Interpretasi:** {interpretation['description']}")

            # Probabilitas prediksi (jika model mendukung)
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(input_scaled)
                    prob_no = proba[0][0] * 100
                    prob_yes = proba[0][1] * 100
                    st.write("### 🔢 Probabilitas Prediksi")
                    st.write(f"• Tidak Diabetes: **{prob_no:.2f}%**")
                    st.write(f"• Diabetes: **{prob_yes:.2f}%**")

                    # Tingkat kepercayaan model
                    max_proba = max(prob_no, prob_yes)
                    if max_proba >= 80:
                        confidence_text, confidence_emoji = "Tinggi", "🟢"
                    elif max_proba >= 60:
                        confidence_text, confidence_emoji = "Sedang", "🟡"
                    else:
                        confidence_text, confidence_emoji = "Rendah", "🔴"
                    st.write(
                        f"**Tingkat Kepercayaan Model:** {confidence_emoji} "
                        f"**{confidence_text}** ({max_proba:.2f}%)"
                    )
                except Exception:
                    pass

            # Deskripsi model yang digunakan
            st.write("### ℹ️ Tentang Model yang Digunakan")
            st.info(f"**{model_option}**: {MODEL_DESCRIPTIONS[model_option]}")

            # Disclaimer
            st.caption(
                "⚠️ *Hasil prediksi ini hanya sebagai alat bantu dan "
                "**bukan merupakan diagnosis medis**. Silakan konsultasikan "
                "dengan dokter untuk diagnosis yang akurat.*"
            )

# ============================================================
# TAB 2: PREDIKSI BATCH (CSV)
# ============================================================
with tab2:
    st.subheader("📂 Prediksi Batch dari File CSV")
    st.write(
        "Upload file CSV yang berisi data pasien untuk memprediksi "
        "semua instance sekaligus."
    )

    st.info(
        "**Format CSV yang diharapkan:**\n\n"
        "File harus memiliki 8 kolom fitur dengan nama persis: "
        "`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, "
        "`Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`."
    )

    # Tombol download template CSV (kosong, hanya header)
    template_df = pd.DataFrame(columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])
    template_csv = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📋 Download Template CSV (kosong)",
        data=template_csv,
        file_name="template_diabetes.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Baca CSV
            df_uploaded = pd.read_csv(uploaded_file)

            # Daftar kolom wajib (urutan sesuai training)
            required_cols = [
                "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
            ]

            # Cek kolom yang hilang
            missing_cols = [c for c in required_cols if c not in df_uploaded.columns]

            if missing_cols:
                st.error(
                    f"❌ Kolom wajib berikut **tidak ditemukan** di CSV: "
                    f"**{', '.join(missing_cols)}**\n\n"
                    f"Pastikan nama kolom sesuai (case-sensitive)."
                )
            else:
                st.success(
                    f"✅ File berhasil diupload: **{len(df_uploaded)} baris data**"
                )

                # Tampilkan data asli
                with st.expander("📋 Lihat Data Asli"):
                    st.dataframe(df_uploaded, use_container_width=True)

                # Ambil hanya kolom fitur dengan urutan yang benar
                input_df = df_uploaded[required_cols].copy()

                # Ganti 0 dengan NaN (sama seperti single prediction)
                cols_zero_to_nan = [
                    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
                ]
                input_df[cols_zero_to_nan] = input_df[cols_zero_to_nan].replace(0, np.nan)

                # Imputasi
                try:
                    input_imputed = imputer.transform(input_df)
                except Exception as e:
                    st.error(f"❌ Gagal saat proses imputasi: {e}")
                    st.stop()

                # Scaling
                try:
                    input_scaled = scaler.transform(input_imputed)
                except Exception as e:
                    st.error(f"❌ Gagal saat proses scaling: {e}")
                    st.stop()

                # Prediksi
                try:
                    predictions = model.predict(input_scaled)
                except Exception as e:
                    st.error(f"❌ Gagal saat melakukan prediksi: {e}")
                    st.stop()

                # Probabilitas (jika model mendukung)
                proba_available = False
                if hasattr(model, "predict_proba"):
                    try:
                        probas = model.predict_proba(input_scaled)
                        proba_available = True
                    except Exception:
                        pass

                # Bangun dataframe hasil
                results_df = df_uploaded.copy()
                results_df["Prediksi"] = [
                    "Tidak Diabetes" if p == 0 else "Diabetes"
                    for p in predictions
                ]

                if proba_available:
                    results_df["Probabilitas Tidak Diabetes (%)"] = (
                        probas[:, 0] * 100
                    ).round(2)
                    results_df["Probabilitas Diabetes (%)"] = (
                        probas[:, 1] * 100
                    ).round(2)

                # Tampilkan tabel hasil
                st.write("---")
                st.subheader("📊 Hasil Prediksi Batch")
                st.dataframe(results_df, use_container_width=True)

                # Ringkasan jumlah
                st.write("### 📈 Ringkasan")
                n_total = len(predictions)
                n_tidak = int((predictions == 0).sum())
                n_diabetes = int((predictions == 1).sum())

                col1, col2, col3 = st.columns(3)
                col1.metric("Total Data", n_total)
                col2.metric("✅ Tidak Diabetes", n_tidak)
                col3.metric("⚠️ Diabetes", n_diabetes)

                # Bar chart sederhana (built-in Streamlit)
                st.write("### 📊 Visualisasi Distribusi")
                chart_data = pd.DataFrame(
                    {
                        "Kategori": ["Tidak Diabetes", "Diabetes"],
                        "Jumlah": [n_tidak, n_diabetes],
                    }
                ).set_index("Kategori")
                st.bar_chart(chart_data)

                # Evaluasi tambahan jika ada kolom label asli
                outcome_aliases = [
                    "Outcome", "outcome", "Label", "label",
                    "Target", "target", "Diabetes", "diabetes",
                ]
                actual_col = None
                for col in outcome_aliases:
                    if col in df_uploaded.columns:
                        actual_col = col
                        break

                if actual_col is not None:
                    st.write(f"### 🎯 Evaluasi (terhadap kolom `{actual_col}`)")
                    y_true = df_uploaded[actual_col].values
                    y_pred = predictions
                    accuracy = (y_true == y_pred).mean() * 100
                    n_benar = int((y_true == y_pred).sum())
                    n_salah = int((y_true != y_pred).sum())

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Akurasi", f"{accuracy:.2f}%")
                    col2.metric("Prediksi Benar", n_benar)
                    col3.metric("Prediksi Salah", n_salah)

                # Tombol download hasil
                st.write("### 💾 Download Hasil")
                csv_result = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Hasil sebagai CSV",
                    data=csv_result,
                    file_name="hasil_prediksi_diabetes.csv",
                    mime="text/csv",
                )

                # Deskripsi model
                st.write("### ℹ️ Tentang Model yang Digunakan")
                st.info(f"**{model_option}**: {MODEL_DESCRIPTIONS[model_option]}")

                # Disclaimer
                st.caption(
                    "⚠️ *Hasil prediksi ini hanya sebagai alat bantu dan "
                    "**bukan merupakan diagnosis medis**.*"
                )
        except Exception as e:
            st.error(f"❌ Gagal membaca atau memproses file CSV: {e}")
