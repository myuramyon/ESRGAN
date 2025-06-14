import streamlit as st
import cv2
import tempfile
from src.main import enhance_image

st.title("🧠 CPU-Based FSRCNN Image Enhancer (5x Zoom)")

uploaded_file = st.file_uploader("Upload a blurry/dark/low-res image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.image(tmp_path, caption="Original Image", use_column_width=True)

    result = enhance_image(tmp_path)
    st.image(result, caption="Enhanced Image (5x Super-Resolution)", use_column_width=True)
