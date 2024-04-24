# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:21:56 2024

@author: Admin
"""


import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Define the directories containing the images
directory_with_haras = r"D:\AI course\Deep learning\harassment detection\dataset\with_haras"
directory_without_haras = r"D:\AI course\Deep learning\harassment detection\dataset\without_haras"

# Function to load and resize images from a directory
def load_images_from_dir(directory, target_size=(64, 64)):
    images_list = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize the image to the target size
                img = cv2.resize(img, target_size)
                images_list.append(img)
            else:
                print(f"Warning: Unable to load image '{img_path}'")
    return images_list


# Load images from both directories
images_with_haras = load_images_from_dir(directory_with_haras)
images_without_haras = load_images_from_dir(directory_without_haras)

# Concatenate the images and create labels
images = images_with_haras + images_without_haras
labels = [1] * len(images_with_haras) + [0] * len(images_without_haras)

# Convert images and labels to numpy arrays
X = np.array(images) / 255.0
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the model
model.save("harassment_detection_model.h5")

