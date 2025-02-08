import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import urllib.request
import os



import gdown
import tensorflow as tf

# Google Drive file ID (replace with your actual ID)
file_id = "https://drive.google.com/file/d/16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo/view?usp=sharing"
url = f"https://drive.google.com/uc?id={file_id}"

# Download the model
output = "biodegradable_classifier.keras"
gdown.download(url, output, quiet=False)

# Load the model
model = tf.keras.models.load_model(output)

# Now you can use the model for predictions





# GitHub Model URL
MODEL_URL = "https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME/raw/main/biodegradable_classifier.keras"

# Download model if not exists
MODEL_PATH = "biodegradable_classifier.keras"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Streamlit UI
st.title("♻️ Biodegradable Image Classifier")
st.write("Upload an image to check if it's biodegradable or non-biodegradable.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Prediction
    prediction = model.predict(img_array)
    class_label = "Biodegradable ✅" if prediction[0][0] < 0.5 else "Non-Biodegradable ❌"

    # Show result
    st.write("### Prediction:")
    st.write(f"**{class_label}**")
