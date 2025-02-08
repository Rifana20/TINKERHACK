import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
import os

# -------------------------------
# ✅ Download and Load Model from Google Drive
# -------------------------------
https://drive.google.com/file/d/16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo/view?usp=sharing
# Correct Google Drive File ID (Extract from your link)
file_id = "16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo"  # ✅ CORRECT (only the file ID)
 
url = f"https://drive.google.com/uc?id={file_id}"

# Define Model Path
MODEL_PATH = "biodegradable_classifier.keras"

# Download Model if Not Exists
if not os.path.exists(MODEL_PATH):
    gdown.download(url, MODEL_PATH, quiet=False)

# Load Model
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# ✅ Streamlit UI
# -------------------------------
st.title("♻️ Biodegradable Image Classifier")
st.write("Upload an image to check if it's biodegradable or non-biodegradable.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    class_label = "Biodegradable ✅" if prediction[0][0] < 0.5 else "Non-Biodegradable ❌"

    # Show result
    st.write("### Prediction:")
    st.write(f"**{class_label}**")  
