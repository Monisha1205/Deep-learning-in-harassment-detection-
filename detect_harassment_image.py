# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 12:31:57 2024

@author: Admin
"""

import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Function to predict user input image
def predict_input_image(image_path):
    # Load the trained model
    model = load_model("harassment_detection_model.h5")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image '{image_path}'")
        return "Prediction failed"

    img = cv2.resize(img, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    if prediction[0][0] > 0.5:
        return "Harassment detected"
    else:
        return "Harassment not detected"

# Example usage:
image_path = r"D:\AI course\Deep learning\harassment detection\dataset\with_haras\angry-male-hand-holding-woman-260nw-1941745984.jpg"
result = predict_input_image(image_path)
print(result)

