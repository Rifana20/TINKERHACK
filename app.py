import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# -------------------------------
# 🔹 Page Configuration
# -------------------------------
st.set_page_config(page_title="♻️ Biodegradable Classifier", page_icon="🌍", layout="wide")

# -------------------------------
# 🔹 Header Section
# -------------------------------
st.markdown(
    """
    <h1 style="text-align: center; color: #4CAF50;">♻️ Biodegradable Image Classifier</h1>
    <p style="text-align: center;">Upload an image to check if it's biodegradable or non-biodegradable.</p>
    <hr style="border: 1px solid #4CAF50;">
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔹 Load the Model from Google Drive
# -------------------------------
file_id = "16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo"  # ✅ Replace with your file ID
url = f"https://drive.google.com/uc?id={file_id}"
MODEL_PATH = "biodegradable_classifier.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model... This may take a few moments."):
        gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load the Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

# -------------------------------
# 🔹 File Uploader with Enhanced UI
# -------------------------------
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # Layout: 2 Columns
    col1, col2 = st.columns(2)

    # Display uploaded image
    with col1:
        st.image(uploaded_file, caption="🖼 Uploaded Image", use_container_width=True, output_format="auto")

    # -------------------------------
    # 🔹 Preprocess and Predict
    # -------------------------------
    with col2:
        with st.spinner("🔄 Analyzing Image..."):
            # Preprocess the image
            img = image.load_img(uploaded_file, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Prediction
            prediction = model.predict(img_array)[0][0]

            # Classify
            if prediction < 0.5:
                st.success("✅ **Biodegradable** ♻️")
                st.markdown("<p style='color:green;'>This object is biodegradable and environmentally friendly!</p>", unsafe_allow_html=True)
            else:
                st.error("❌ **Non-Biodegradable** 🚯")
                st.markdown("<p style='color:red;'>This object is non-biodegradable and may harm the environment.</p>", unsafe_allow_html=True)

            # Progress Bar Effect
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown(
    """
    <hr>
    <p style="text-align: center;">Developed with ❤️ using Streamlit & TensorFlow | <a href="https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME" target="_blank">View on GitHub</a></p>
    """,
    unsafe_allow_html=True
)
