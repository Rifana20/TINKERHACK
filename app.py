import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image

# -------------------------------
# 🔹 Page Configuration & Styling
# -------------------------------
st.set_page_config(
    page_title="♻️ Biodegradable Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Background & Styling
st.markdown(
    """
    <style>
    /* Background Image */
    .stApp {
        background: url("https://source.unsplash.com/1600x900/?nature,green") no-repeat center fixed;
        background-size: cover;
    }
    
    /* Heading Design */
    .title {
        font-size: 42px;
        text-align: center;
        color: #fff;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }

    .subtitle {
        font-size: 22px;
        text-align: center;
        color: #fff;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        font-size: 16px;
        color: #fff;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔹 Page Title
# -------------------------------
st.markdown('<p class="title">♻️ Biodegradable Image Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to check if it is Biodegradable or Non-Biodegradable</p>', unsafe_allow_html=True)

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

                # Display Result
                if prediction < 0.5:
                    st.success("✅ **Biodegradable** ♻️")
                else:
                    st.error("❌ **Non-Biodegradable** 🚯")

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
        <a href="https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME" target="_blank" style="color:#fff; text-decoration:none;">View on GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
