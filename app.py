import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -------------------------------
# Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("covid_xray_cnn.h5")

if not os.path.exists("covid_xray_cnn.h5"):
    st.error("Model file not found. Please place 'covid_xray_cnn.h5' in the app directory.")
    st.stop()

model = load_model()

# -------------------------------
# App UI
# -------------------------------
st.title("COVID-19 X-Ray Detection System")
st.write("Upload a chest X-ray image to classify it as COVID or Normal.")

# -------------------------------
# File upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Display resized image (UI purpose only)
    display_image = image.resize((300, 300))
    st.image(display_image, caption="Uploaded X-ray")

    # Preprocess image for model
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # # Prediction
    prediction = model.predict(img_array)[0][0]
    
if prediction < 0.5:
    label = "COVID"
    confidence = (1 - prediction) * 100
else:
    label = "Normal"
    confidence = prediction * 100

st.markdown(
    f"""
    <div style="margin-top:15px;">
        <p style="font-size:18px; font-weight:600;">
            Prediction: <span style="font-weight:700;">{label}</span>
        </p>
        <p style="font-size:16px; color:#cfcfcf;">
            Confidence: <span style="font-weight:700;">{confidence:.2f}%</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
