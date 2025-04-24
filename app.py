# Import libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import os

# Load the trained model
model_path = "arabic_sign_language_model.h5"  # Update to your model path
try:
    model = load_model(model_path)
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set up MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

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

    # Flip image if left hand detected
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

# Video processor class for streamlit-webrtc
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_letter = "None"
        self.current_confidence = 0.0
        self.sentence = st.session_state.get("sentence", "")

    def update_sentence(self):
        st.session_state.sentence = self.sentence

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr")
        img = cv2.flip(img, 1)  # Flip frame

        # Extract landmarks
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
            cv2.putText(img_processed, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw landmarks
        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img_processed, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img_processed, format="bgr")

# Streamlit app
def main():
    # Custom CSS for modern UI
    st.markdown(
        """
        <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            max-width: 1200px;
            margin: auto;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .header {
            color: #2c3e50;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        .webcam-container {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            background: #ecf0f1;
        }
        .prediction-box {
            font-size: 1.5em;
            color: #2c3e50;
            background: #e8f4f8;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .sentence-box {
            font-size: 1.2em;
            color: #34495e;
            background: #dfe6e9;
            padding: 15px;
            border-radius: 8px;
            min-height: 50px;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1.1em;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #2980b9;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Initialize session state
    if "sentence" not in st.session_state:
        st.session_state.sentence = ""

    # Header
    st.markdown("<h1 class='header'>SilenTalker: Arabic Sign Language Recognition</h1>", unsafe_allow_html=True)
    st.markdown("Capture hand gestures via webcam to recognize Arabic letters and build sentences.")

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='webcam-container'>", unsafe_allow_html=True)
        webrtc_ctx = webrtc_streamer(
            key="sign-language",
            video_processor_factory=SignLanguageProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        processor = webrtc_ctx.video_processor if webrtc_ctx and webrtc_ctx.video_processor else None
        prediction = "None (0.00%)"
        if processor:
            prediction = f"{processor.current_letter} ({processor.current_confidence:.2f}%)"
        st.markdown(f"<div class='prediction-box'>Predicted Letter: {prediction}</div>", unsafe_allow_html=True)

        # Buttons
        if st.button("Add Letter"):
            if processor and processor.current_letter != "None" and processor.current_confidence > 80:
                processor.sentence += processor.current_letter
                processor.update_sentence()

        if st.button("Backspace"):
            if processor and processor.sentence:
                processor.sentence = processor.sentence[:-1]
                processor.update_sentence()

        if st.button("Add Space"):
            if processor:
                processor.sentence += " "
                processor.update_sentence()

    # Sentence output
    st.markdown(f"<div class='sentence-box'>Sentence: {st.session_state.sentence}</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='footer'>Developed by LINK Team</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
