import streamlit as st
import requests
from PIL import Image
from io import BytesIO


st.title("Image Classification")
st.write("Upload an image to classify")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        res = requests.post("http://localhost:5000/classify", files={"file": uploaded_file.getvalue()})
        class_result = res.json()
        st.write("Prediction:", class_result["classification"])
