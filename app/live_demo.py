import streamlit as st
import cv2
import numpy as np
from src.capture import get_webcam_frame
from src.preprocessing import preprocess_frame
from src.prediction import predict
from src.text_to_speech import speak
from PIL import Image

st.title("Sign Language to Speech (A-G)")

mode = st.radio("Select Input Mode", ["Webcam (Local Only)", "Upload Image"])

if mode == "Webcam (Local Only)":
    st.info("Webcam works only locally, not on Streamlit Cloud.")
    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot access webcam")
        else:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if st.button("Predict Gesture"):
                    processed = preprocess_frame(frame)
                    letter = predict(processed)
                    st.success(f"Predicted Letter: {letter}")
                    speak(letter)
            cap.release()

else:  # Upload mode for cloud
    uploaded_file = st.file_uploader("Upload a hand gesture image", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        img_array = np.array(image)
        processed = preprocess_frame(img_array)
        letter = predict(processed)
        st.success(f"Predicted Letter: {letter}")
        speak(letter)
