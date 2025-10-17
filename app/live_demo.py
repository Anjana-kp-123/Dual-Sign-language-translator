import streamlit as st
import numpy as np
from PIL import Image
from src.preprocessing import preprocess_frame
from src.prediction import predict
from src.text_to_speech import speak

st.title("Sign Language to Speech (A-G)")

uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    processed = preprocess_frame(img_array)
    letter = predict(processed)
    st.success(f"Predicted Letter: {letter}")
    speak(letter)
