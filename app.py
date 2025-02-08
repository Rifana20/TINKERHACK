import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

# ✅ Correct Google Drive File ID
file_id = "16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo"
url = f"https://drive.google.com/uc?id={file_id}"
MODEL_PATH = "biodegradable_classifier.keras"

# ✅ Download model only if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.write("Downloading Model... Please wait.")
    gdown.download(url, MODEL_PATH, quiet=False)

# ✅ Load the Model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.write("❌ Error Loading Model:", e)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("♻️ Biodegradable Image Classifier")
st.write("Upload an image to check if it's biodegradable or non-biodegradable.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # ✅ FIXED DEPRECATION WARNING

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_label = "Biodegradable ✅" if prediction[0][0] < 0.5 else "Non-Biodegradable ❌"

    # Show result
    st.write("### Prediction:")
    st.write(f"**{class_label}**")
