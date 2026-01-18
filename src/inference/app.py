import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "models/vehicle10_mobilenet_best.keras"
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    'bicycle', 'boat', 'bus', 'car', 'helicopter',
    'minibus', 'motorcycle', 'taxi', 'train', 'truck'
]

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Vehicle Classifier", layout="centered")
st.title("Vehicle Image Classification")
st.write("Upload an image and the model will predict the vehicle type.")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction logic
# -------------------------------
def preprocess_image(image: Image.Image):
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=700)

    img_tensor = preprocess_image(image)
    preds = model.predict(img_tensor)[0]

    pred_index = np.argmax(preds)
    confidence = preds[pred_index]

    st.subheader("Prediction")
    st.markdown(f"""
    **Vehicle Type:** `{CLASS_NAMES[pred_index]}`  
    **Confidence:** `{confidence:.2%}`
    """)

    st.progress(float(confidence))
