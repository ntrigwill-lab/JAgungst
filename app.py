import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from keras.layers import TFSMLayer
from keras import Input, Model
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess_input

# Mapping class
map_class = {
    0: 'Northern Leaf Blight',
    1: 'Common Rust',
    2: 'Gray Leaf Spot',
    3: 'Healthy'
}

# Load model menggunakan TFSMLayer (SavedModel .pb inference-only)
@st.cache_resource
def load_model():
    base_layer = TFSMLayer("model_mobnetv2_inference", call_endpoint="serving_default")
    inputs = Input(shape=(224, 224, 3))
    outputs = base_layer(inputs)
    model = Model(inputs, outputs)
    return model

# Fungsi prediksi gambar
def predict_image(img: Image.Image):
    model = load_model()
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = mobilenet_preprocess_input(img_array)

    preds = model.predict(img_array)

    if isinstance(preds, dict):
        return list(preds.values())[0][0]
    else:
        return preds[0]

# UI Streamlit
st.title("🌽 Deteksi Penyakit Daun Jagung")
st.markdown("Upload gambar daun jagung untuk mendeteksi penyakit menggunakan model MobileNetV2.")

uploaded_file = st.file_uploader("🖼️ Upload gambar daun", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar diupload", use_column_width=True)

    with st.spinner("🔍 Memprediksi..."):
        preds = predict_image(image)
        df = pd.DataFrame([preds], columns=map_class.values())
        st.subheader("📊 Hasil Prediksi")
        st.dataframe(df)

        pred_class = np.argmax(preds)
        label = map_class[pred_class]

        if label == "Healthy":
            st.success(f"✅ Daun Jagung Sehat")
        else:
            st.error(f"⚠️ Daun Jagung Terkena Penyakit: {label}")