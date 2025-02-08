import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image
import base64

# -------------------------------
# 🔹 Page Configuration & Styling
# -------------------------------
st.set_page_config(
    page_title="♻️ Biodegradable Image Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 🔹 Function to Set Background Image
# -------------------------------
def set_background(image_url):
    """Encodes an image as base64 and sets it as a background in Streamlit."""
    with open(image_url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# ✅ Load Background Image (Ensure 'background.webp' is in the same folder)
set_background("background.webp")

# -------------------------------
# 🔹 Custom Styling for Headings
# -------------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 80px;  /* Even Bigger */
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 5px 5px 12px rgba(0, 0, 0, 0.9);
    }
    
    .subtitle {
        font-size: 40px;  /* Bigger Subtitle */
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 4px 4px 10px rgba(0, 0, 0, 0.9);
    }

    .prediction-box {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 28px;
        font-weight: bold;
        box-shadow: 5px 5px 12px rgba(0, 0, 0, 0.8);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔹 Page Title (Now Even Bigger)
# -------------------------------
st.markdown('<p class="title">♻️ Biodegradable Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📸 Upload an image to check if it is Biodegradable or Non-Biodegradable</p>', unsafe_allow_html=True)

# -------------------------------
# 🔹 Load Model from Google Drive
# -------------------------------
file_id = "16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo"  # ✅ Replace with your actual Google Drive file ID
url = f"https://drive.google.com/uc?id={file_id}"
MODEL_PATH = "biodegradable_classifier.keras"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading model... Please wait."):
        try:
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("✅ Model Downloaded Successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

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
    col1, col2 = st.columns([1, 2])

    # Display uploaded image
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", use_container_width=True)  # ✅ Fixed Warning

    # -------------------------------
    # 🔹 Preprocess and Predict
    # -------------------------------
    with col2:
        st.subheader("🔍 Model Prediction")
        with st.spinner("🔄 Analyzing Image..."):
            try:
                # Preprocess the image
                img = img.resize((150, 150))
                img_array = np.array(img) / 255.0  # Normalize
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                prediction = model.predict(img_array)[0][0]

                # Display Result Inside a Styled Box
                result = "✅ **Biodegradable** ♻️" if prediction < 0.5 else "❌ **Non-Biodegradable** 🚯"
                st.markdown(f'<div class="prediction-box">{result}</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <p class="footer">
        Developed with ❤️ using Streamlit & TensorFlow | 
        <a href="https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME" target="_blank" style="color:#ffffff; text-decoration:none;">View on GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
