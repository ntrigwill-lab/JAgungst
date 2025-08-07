import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

# Fungsi untuk preprocessing gambar secara aman
def preprocess_image(img: Image.Image):
    # Pastikan gambar dalam format RGB (bukan RGBA)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Load model SavedModel (.pb format) - harus folder, bukan file tunggal
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_mobnetv2", compile=False)
    return model

# Mapping label kelas
map_class = {
    0: 'Northern Leaf Blight',
    1: 'Common Rust',
    2: 'Gray Leaf Spot',
    3: 'Healthy'
}

# Fungsi prediksi
def predict_image(img: Image.Image):
    model = load_model()
    img = preprocess_image(img)
    preds = model.predict(img)
    return preds[0]

# Streamlit UI
st.title("🌽 Klasifikasi Penyakit Daun Jagung")
uploaded_file = st.file_uploader("🖼️ Upload gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    try:
        preds = predict_image(image)

        # Tampilkan hasil prediksi dalam bentuk DataFrame
        df_results = pd.DataFrame([preds], columns=[
            'Northern Leaf Blight', 'Common Rust', 'Gray Leaf Spot', 'Healthy'
        ])
        st.subheader("📊 Hasil Prediksi:")
        st.dataframe(df_results)

        # Tampilkan hasil final
        predicted_class = np.argmax(preds)
        result = map_class[predicted_class]
        st.success(f"Hasil: Daun Jagung terdeteksi sebagai **{result}**")

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")
