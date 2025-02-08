import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown
from PIL import Image
import base64
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------------
# 🔹 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="♻️ EcoVision: AI Waste Classifier",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------
# 🔹 Function to Set Background
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
    .title {{
        font-size: 60px !important;
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
    }}
    .subtitle {{
        font-size: 28px !important;
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

set_background("background.webp")

# -------------------------------
# 🔹 Page Title
# -------------------------------
st.markdown('<p class="title">♻️ EcoVision: AI Waste Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image to check if it is Biodegradable or Non-Biodegradable</p>', unsafe_allow_html=True)

# -------------------------------
# 🔹 Load Model from Google Drive
# -------------------------------
file_id = "16bvig8rIuZqeahGDr3S38Dt1ZIAnNJIo"
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

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error Loading Model: {e}")
    st.stop()

# -------------------------------
# 🔹 Image Uploader
# -------------------------------
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Model Prediction")
        container = st.container()
        
        with container:
            with st.spinner("🔄 Analyzing Image..."):
                try:
                    img = img.resize((150, 150))
                    img_array = np.array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    prediction = model.predict(img_array)[0][0]

                    if prediction < 0.5:
                        st.success("✅ **Biodegradable** ♻️")
                    else:
                        st.error("❌ **Non-Biodegradable** 🚯")

                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")

# -------------------------------
# 🔹 Chatbot Section (Mistral-7B)
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("🤖 AI Chatbot: Ask me anything!")

# 🔹 Load Mistral-7B Model from Hugging Face
@st.cache_resource()  # Cache to avoid reloading
def load_mistral_model():
    model_name = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chat_model = load_mistral_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display past messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about biodegradable waste...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_model(user_input, max_length=200, do_sample=True)
            reply = response[0]['generated_text']
            st.markdown(reply)

    st.session_state["messages"].append({"role": "assistant", "content": reply})

# -------------------------------
# 🔹 Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <p style="text-align: center; font-size: 18px;">
        Developed with ❤️ using Streamlit & TensorFlow | 
        <a href="https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME" target="_blank">View on GitHub</a>
    </p>
    """,
    unsafe_allow_html=True
)
