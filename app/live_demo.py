import streamlit as st
import cv2
from src.capture import get_webcam_frame
from src.preprocessing import preprocess_frame
from src.prediction import predict
from src.text_to_speech import speak

st.title("Sign Language to Speech (A-G)")

run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
            break

        frame = cv2.flip(frame, 1)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if st.button("Predict Gesture"):
            processed = preprocess_frame(frame)
            letter = predict(processed)
            st.success(f"Predicted Letter: {letter}")
            speak(letter)
