import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from PIL import Image

# -------------------------------
# 🔹 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="♻️ Biodegradable Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown(
    """
    <style>
        body {
            background-color: #F5F5F5;
        }
        .main-title {
            text-align: center;
            font-size: 36px;
            color: #2E7D32;
            font-weight: bold;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .st-emotion-cache-0 {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 🔹 Sidebar
# -------------------------------
with st.sidebar:
    st.image("https://i.imgur.com/OK7R6XL.png", use_column_width=True)  # Replace with your logo if needed
    st.markdown("### 🌱 About the Project")
    st.write(
        "This AI model predicts whether an object is **biodegradable** or **non-biodegradable**. "
        "Upload an image and get an instant classification!"
    )
    st.info("📌 Model trained using TensorFlow & CNNs.")
    st.markdown("<hr>", unsafe_allow_html=True)

# -------------------------------
# 🔹 Header Section
# -------------------------------
st.markdown('<h1 class="main-title">♻️ Biodegradable Image Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image to check if it’s biodegradable or non-biodegradable.</p>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

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
        st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

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
                    st.markdown(
                        "<p style='color:green; font-size:18px;'>This object is biodegradable and environmentally friendly!</p>",
                        unsafe_allow_html=True
                    )
                    st.balloons()
                else:
                    st.error("❌ **Non-Biodegradable** 🚯")
                    st.markdown(
                        "<p style='color:red; font-size:18px;'>This object is non-biodegradable and may harm the environment.</p>",
                        unsafe_allow_html=True
                    )
                    st.snow()

                # Animated Progress Bar
                progress = st.progress(0)
                for i in range(100):
                    progress.progress(i + 1)

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <p style="text-align: center; font-size: 16px;">
        Developed with ❤️ using Streamlit & TensorFlow | 
        <a href="https://github.com/Rifana20/TINKERHACK" target="_blank">View on GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
