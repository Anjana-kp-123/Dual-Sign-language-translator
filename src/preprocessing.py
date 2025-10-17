from PIL import Image
import numpy as np

IMG_SIZE = 64

def preprocess_frame(frame):
    """
    frame: numpy array or PIL image
    Returns a 64x64 grayscale image as numpy array ready for model
    """
    # Convert to PIL Image if it's numpy array
    if isinstance(frame, np.ndarray):
        frame = Image.fromarray(frame)

    # Convert to grayscale
    frame = frame.convert("L")

    # Resize
    frame = frame.resize((IMG_SIZE, IMG_SIZE))

    # Normalize
    frame_array = np.array(frame) / 255.0

    # Expand dims for model (batch_size, height, width, channels)
    return np.expand_dims(frame_array, axis=(0, -1))
