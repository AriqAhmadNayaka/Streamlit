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

        if prediction[0] == 0:
            st.success("✅ Hasil Prediksi: **Tidak Diabetes**")
        else:
            st.error("⚠️ Hasil Prediksi: **Diabetes**")

        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(input_scaled)
                prob_no = proba[0][0] * 100
                prob_yes = proba[0][1] * 100
                st.write("### 🔢 Probabilitas Prediksi")
                st.write(f"• Tidak Diabetes: **{prob_no:.2f}%**")
                st.write(f"• Diabetes: **{prob_yes:.2f}%**")
            except Exception:
                pass

        st.caption(f"Model yang digunakan: **{model_option}**")
