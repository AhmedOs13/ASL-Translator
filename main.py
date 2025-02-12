import av
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import gdown
import os

# Model configuration
MODEL_URL = "https://drive.google.com/uc?export=download&id=17BSlxnvZMWrAf3g1l1Mn0pWkDo4PoOn9"
MODEL_PATH = "model_alphabet_transfer.keras"

# Model loading with caching
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH, compile=False)

# Load the trained model
model = load_my_model()

# Define class labels
class_labels = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit UI
st.title("🤟 Real-time ASL Gesture Recognition")
st.write("This application uses a trained model to recognize ASL hand gestures in real time.")

# Define video processing class
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert to numpy array

        # Resize frame to match model input size
        img_resized = cv2.resize(img, (256, 256))  
        img_array = np.expand_dims(img_resized, axis=0)  
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        predicted_label = class_labels[predicted_class]

        # Display prediction on frame
        label_text = f'{predicted_label} ({confidence:.2f})'
        cv2.putText(img, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start webcam streaming
webrtc_streamer(key="asl-gesture", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
