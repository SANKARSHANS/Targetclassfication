import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import csv
import random

# Load or create a pre-trained model
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)  # 3 classes: Bird, Drone, Nothing
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_model()

# Load weights if available
# Uncomment and use your fine-tuned weights
# model.load_weights("fine_tuned_weights.h5")

# Define class labels
CLASS_LABELS = ["Bird", "Drone", "Nothing"]

FRAME_SIZE = (224, 224)

# Function to preprocess video frames
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, FRAME_SIZE)
    frame_preprocessed = preprocess_input(frame_resized)
    return frame_preprocessed

# Function to generate CSV file
def generate_csv(file_name, frequency_range):
    frequencies = [random.uniform(*frequency_range) for _ in range(30)]
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency"])
        writer.writerows([[freq] for freq in frequencies])

# Function to classify video
def classify_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

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

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Generate CSV file based on classification
    if detected_class == "Drone":
        generate_csv("microdopllersignals.csv", (-45, 45))
    elif detected_class == "Bird":
        generate_csv("microdopllersignals.csv", (-0.45, 0.45))
    else:
        generate_csv("microdopllersignals.csv", (-100, 100))

    print(f"Detected Class: {detected_class}")

if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")
    classify_video(video_path)
