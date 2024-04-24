# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 16:11:35 2024

@author: Admin
"""


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the trained model
model = load_model("harassment_detection_model.h5")

# Function to preprocess an image
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))  # Resize image to match model input size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict whether an image contains harassment
def predict_image(img):
    img = preprocess_image(img)
    prediction = model.predict(img)[0][0]
    return prediction

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize parameters
threshold = 0.5  # Initial prediction threshold
scale_factor = 0.1  # Factor by which to adjust threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame
    preprocessed_img = preprocess_image(frame)

    # Predict on the preprocessed frame
    confidence = predict_image(frame)

    # Apply threshold to determine prediction label
    if confidence >= threshold:
        label = "With Harassment"
        color = (0, 0, 255)  # Red color for "With Harassment"
    else:
        label = "Without Harassment"
        color = (0, 255, 0)  # Green color for "Without Harassment"

    # Display the label and bounding box on the captured frame
    cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), color, 3)

    # Display the captured frame
    cv2.imshow('Webcam', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
