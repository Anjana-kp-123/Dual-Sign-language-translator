from tensorflow.keras.models import load_model
import numpy as np
import os

model_path = "models/small_cnn.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Run scripts/init_model.py first.")

model = load_model(model_path)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def predict(frame):
    pred = model.predict(frame)
    idx = np.argmax(pred)
    return labels[idx]
