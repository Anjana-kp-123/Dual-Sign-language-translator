import cv2

def get_webcam_frame():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible")
    
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Failed to capture frame")
    
    cap.release()
    frame = cv2.flip(frame, 1)
    return frame
