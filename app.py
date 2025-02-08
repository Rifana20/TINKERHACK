# -------------------------------
# 🔹 Custom Styling for Headings
# -------------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 60px;  /* Increased size */
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.8);
    }
    
    .subtitle {
        font-size: 30px;  /* Increased size */
        text-align: center;
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.8);
    }
    
    .footer {
        text-align: center;
        font-size: 18px;
        color: #ffffff;
        margin-top: 30px;
        font-weight: bold;
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
