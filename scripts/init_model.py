import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import os

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

os.makedirs("models", exist_ok=True)
model.save("models/small_cnn.h5")

dummy_input = np.random.rand(1,64,64,1)
pred = model.predict(dummy_input)
print("Dummy model initialized. Sample prediction:", labels[np.argmax(pred)])
