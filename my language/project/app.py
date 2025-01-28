import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import csv
import random
import os

# Function to create a model
@st.cache_resource
def create_model():
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(3, activation="softmax")(x)  # 3 classes: Bird, Drone, Nothing
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model()
CLASS_LABELS = ["Bird", "Drone", "Nothing"]
FRAME_SIZE = (224, 224)

# Preprocess frame
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    frame_preprocessed = preprocess_input(frame_resized)
    return frame_preprocessed

# Generate CSV file
def generate_csv(file_name, frequency_range):
    frequencies = [random.uniform(*frequency_range) for _ in range(30)]
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency"])
        writer.writerows([[freq] for freq in frequencies])

# Video classification
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open video.")
        return "Nothing"

    detected_class = "Nothing"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        prediction = model.predict(np.expand_dims(processed_frame, axis=0))
        predicted_class = CLASS_LABELS[np.argmax(prediction[0])]

        if predicted_class != "Nothing":
            detected_class = predicted_class
            break

    cap.release()

    # Generate CSV based on detected class
    if detected_class == "Drone":
        generate_csv("output_drone.csv", (-45, 45))
        return "Drone", "output_drone.csv"
    elif detected_class == "Bird":
        generate_csv("output_bird.csv", (-0.45, 0.45))
        return "Bird", "output_bird.csv"
    else:
        generate_csv("output_random.csv", (-100, 100))
        return "Nothing", "output_random.csv"

# Classify CSV file with improved logic
def classify_csv(file):
    data = file.read().decode("utf-8").split("\n")[1:]  # Skip the header
    frequencies = [float(line.strip()) for line in data if line.strip()]

    # Calculate average frequency and standard deviation
    avg_frequency = np.mean(frequencies)
    std_dev_frequency = np.std(frequencies)

    # Define thresholds
    if all(-45 <= freq <= 45 for freq in frequencies):
        if avg_frequency > -1 and avg_frequency < 1 and std_dev_frequency < 1:  # Tight range for bird
            return "Bird"
        return "Drone"  # Wider range for drone
    else:
        return "Unclassified"


# Streamlit UI
st.title("Microdoppler Based target Classification")

# Step 1: Video Processing
st.header("Step 1: Upload and Process Video")
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if video_file:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    st.video(video_file)

    st.write("Processing video...")
    detected_class, csv_path = classify_video(video_path)
    st.write("CSV file generated:")
    with open(csv_path, "rb") as f:
        st.download_button("Download CSV", f, file_name=csv_path)

# Step 2: CSV Classification
st.header("Step 2: Upload a CSV file for target classification")
csv_file = st.file_uploader("Upload a CSV file", type=["csv"])
if csv_file:
    result = classify_csv(csv_file)
    st.success(f"CSV classified as: {result}")
