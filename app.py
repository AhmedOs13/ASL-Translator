import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os

# Check for OpenCV installation
try:
    cv2.__version__
    st.success("OpenCV loaded successfully")
except ImportError as e:
    st.error(f"Failed to import OpenCV: {e}")
    st.error("Ensure 'opencv-python-headless' is listed in requirements.txt")
    st.stop()

# Load the trained model
model_path = "arabic_sign_language_model.h5"
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' not found. Please upload it to the app directory.")
    st.stop()

try:
    model = load_model(model_path)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up MediaPipe
try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    st.success("MediaPipe initialized")
except Exception as e:
    st.error(f"Error initializing MediaPipe: {e}")
    st.stop()

# Function to extract and scale landmarks
def extract_and_scale_landmarks(image, img_size=400, padding=50):
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return np.zeros(63), image, False

    landmarks = results.multi_hand_landmarks[0]
    handedness = "Right"
    if results.multi_handedness and len(results.multi_handedness) > 0:
        handedness = results.multi_handedness[0].classification[0].label

    if handedness == "Left":
        image = cv2.flip(image, 1)
        for landmark in landmarks.landmark:
            landmark.x = 1 - landmark.x

    points = [(int(l.x * w), int(l.y * h)) for l in landmarks.landmark]
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    hand_width, hand_height = x_max - x_min, y_max - y_min
    if hand_width < 10 or hand_height < 10:
        return np.zeros(63), image, False

    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

    hand_width, hand_height = x_max - x_min, y_max - y_min
    scale = (img_size - 2 * padding) / max(hand_width, hand_height)
    offset_x = (img_size - hand_width * scale) / 2
    offset_y = (img_size - hand_height * scale) / 2

    scaled_points = [
        (int((x - x_min) * scale + offset_x), int((y - y_min) * scale + offset_y))
        for x, y in points
    ]

    normalized_points = []
    for x, y in scaled_points:
        normalized_points.extend([x / img_size, y / img_size, 0.0])

    return np.array(normalized_points), image, True

# Label mapping
labels = ['ع', 'ال', 'ا', 'ب', 'ض', 'د', 'ف', 'غ', 'ح', 'ه', 'ج', 'ك', 'خ', 'لا', 'ل', 'م', 'ن', 'ق', 'ر', 'ص', 'س',
          'ش', 'ط', 'ت', 'ة', 'ذ', 'ث', 'و', 'ي', 'ظ', 'ز']
label_map = {i: label for i, label in enumerate(labels)}

# Function to process a static image (fallback)
def process_static_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.flip(img, 1)

    landmarks, img_processed, hand_detected = extract_and_scale_landmarks(img)

    if hand_detected:
        landmarks = landmarks.reshape(1, 63)
        prediction = model.predict(landmarks, verbose=0)
        predicted_label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        letter = label_map[predicted_label]
    else:
        letter = "None"
        confidence = 0.0
        cv2.putText(img_processed, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img_processed, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    return img_processed, letter, confidence

# Video processor class for streamlit-webrtc
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_letter = "None"
        self.current_confidence = 0.0
        self.sentence = st.session_state.get("sentence", "")
        self.error = None

    def update_sentence(self):
        st.session_state.sentence = self.sentence

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr")
            img = cv2.flip(img, 1)

            landmarks, img_processed, hand_detected = extract_and_scale_landmarks(img)

            if hand_detected:
                landmarks = landmarks.reshape(1, 63)
                prediction = model.predict(landmarks, verbose=0)
                predicted_label = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100
                self.current_letter = label_map[predicted_label]
                self.current_confidence = confidence
            else:
                self.current_letter = "None"
                self.current_confidence = 0.0
                cv2.putText(img_processed, "No hand detected", (10, 30), cv2.FONT
