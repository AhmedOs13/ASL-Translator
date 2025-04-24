import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tkinter as tk
from PIL import Image, ImageTk
import threading
import queue
from tkinter import ttk

# Load pre-trained models
arabic_model = load_model('arabic_sign_language_model.h5')  # Ensure correct path
english_model = load_model('english_sl_model_v2.h5')  # Ensure correct path

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)


# Extract and scale hand landmarks for model input
def extract_and_scale_landmarks(image, img_size=400, padding=50):
    """
    Extracts hand landmarks from an image, scales, and centers them for model input.

    Args:
        image: Input image (BGR format).
        img_size (int): Target size for scaled landmarks.
        padding (int): Padding around the hand bounding box.

    Returns:
        tuple: (normalized landmarks, processed image, hand detection flag)
    """
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return np.zeros(63), image, False

    landmarks = results.multi_hand_landmarks[0]
    handedness = "Right"
    if results.multi_handedness and len(results.multi_handedness) > 0:
        handedness = results.multi_handedness[0].classification[0].label

    # Flip image if left hand is detected
    if handedness == "Left":
        image = cv2.flip(image, 1)
        for landmark in landmarks.landmark:
            landmark.x = 1 - landmark.x

    # Extract landmark coordinates
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


# Scale the video frame for display
def scale_frame(frame, scale_factor=1.5):
    """
    Scales the input frame by a given factor.

    Args:
        frame: Input image frame.
        scale_factor (float): Scaling factor for resizing.

    Returns:
        Scaled frame.
    """
    height, width = frame.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    return cv2.resize(frame, (new_width, new_height))


# Define label mappings for Arabic and English sign language
arabic_labels = ['ع', 'ال', 'ا', 'ب', 'ض', 'د', 'ف', 'غ', 'ح', 'ه', 'ج', 'ك', 'خ', 'لا', 'ل', 'م', 'ن', 'ق', 'ر', 'ص',
                 'س',
                 'ش', 'ط', 'ت', 'ة', 'ذ', 'ث', 'و', 'ي', 'ظ', 'ز']
english_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U',
                  'V', 'W', 'X', 'Y', 'Z']

arabic_label_map = {i: label for i, label in enumerate(arabic_labels)}
english_label_map = {i: label for i, label in enumerate(english_labels)}


# GUI class for sign language recognition
class SignLanguageGUI:
    def __init__(self, root):
        """
        Initializes the GUI for sign language recognition.

        Args:
            root: Tkinter root window.
        """
        self.root = root
        self.root.title("Sign Language Recognition")
        self.root.geometry("800x700")

        # Initialize variables
        self.sentence = ""
        self.current_letter = ""
        self.current_confidence = 0.0
        self.running = True
        self.frame_queue = queue.Queue(maxsize=1)
        self.current_model = arabic_model
        self.current_label_map = arabic_label_map
        self.model_name = "Arabic"

        # Header
        self.header_label = tk.Label(root, text="SilenTalker", font=("Arial", 24, "bold"))
        self.header_label.pack(pady=10)

        # Model selection dropdown
        self.model_label = tk.Label(root, text="Select Language Model:", font=("Arial", 12))
        self.model_label.pack()
        self.model_var = tk.StringVar(value="Arabic")
        self.model_dropdown = ttk.Combobox(root, textvariable=self.model_var, values=["Arabic", "English"],
                                           state="readonly")
        self.model_dropdown.pack(pady=5)
        self.model_dropdown.bind("<<ComboboxSelected>>", self.switch_model)

        # GUI elements
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10)

        self.letter_label = tk.Label(root, text="Predicted Letter: None", font=("Arial", 14))
        self.letter_label.pack()

        self.sentence_label = tk.Label(root, text="Sentence: ", font=("Arial", 14))
        self.sentence_label.pack(pady=10)

        self.sentence_text = tk.Text(root, height=1, width=50, font=("Arial", 15))
        self.sentence_text.pack()
        self.sentence_text.config(state='disabled')

        # Footer
        self.footer_label = tk.Label(root, text="Developed by Minia University, Biomedical Engineering Department", font=("Arial", 13))
        self.footer_label.pack(pady=10)

        # Key bindings
        self.root.bind('<Return>', self.add_letter)
        self.root.bind('<BackSpace>', self.backspace)
        self.root.bind('<space>', self.add_space)

        # Start video processing in a separate thread
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.letter_label.config(text="Error: Could not open webcam.")
            return

        self.video_thread = threading.Thread(target=self.process_video, daemon=True)
        self.video_thread.start()

        # Start GUI updates
        self.update_gui()

    def switch_model(self, event):
        """Switches between Arabic and English models based on dropdown selection."""
        selected_model = self.model_var.get()
        if selected_model == "Arabic":
            self.current_model = arabic_model
            self.current_label_map = arabic_label_map
            self.model_name = "Arabic"
        else:
            self.current_model = english_model
            self.current_label_map = english_label_map
            self.model_name = "English"
        self.sentence = ""  # Reset sentence on model switch
        self.update_sentence()

    def process_video(self):
        """Processes webcam video to detect and classify hand signs."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Flip frame horizontally
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            landmarks, frame_processed, hand_detected = extract_and_scale_landmarks(frame, img_size=400, padding=50)

            if hand_detected:
                # Predict using the selected model
                landmarks = landmarks.reshape(1, 63)
                prediction = self.current_model.predict(landmarks, verbose=0)
                predicted_label = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction) * 100
                predicted_letter = self.current_label_map[predicted_label]

                self.current_letter = predicted_letter
                self.current_confidence = confidence
            else:
                self.current_letter = "None"
                self.current_confidence = 0.0
                frame_processed = scale_frame(frame_processed, scale_factor=1.5)
                cv2.putText(frame_processed, "No hand detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3,
                            cv2.LINE_AA)

            # Convert frame for GUI display
            frame_processed = scale_frame(frame_processed, scale_factor=1.5)
            frame_rgb = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            image = image.resize((640, 480), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Update frame queue
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(photo)

    def update_gui(self):
        """Updates the GUI with the latest video frame and predictions."""
        try:
            # Update video feed
            if not self.frame_queue.empty():
                photo = self.frame_queue.get()
                self.video_label.config(image=photo)
                self.video_label.image = photo  # Keep reference

            # Update text displays
            letter_text = f"Predicted Letter: {self.current_letter} ({self.current_confidence:.2f}%) [{self.model_name}]"
            self.letter_label.config(text=letter_text)
            self.sentence_text.config(state='normal')
            self.sentence_text.delete(1.0, tk.END)
            self.sentence_text.insert(tk.END, self.sentence)
            self.sentence_text.config(state='disabled')

        except Exception as e:
            print(f"GUI Update Error: {e}")

        # Schedule next update
        if self.running:
            self.root.after(10, self.update_gui)

    def add_letter(self, event):
        """Adds the predicted letter to the sentence if confidence is sufficient."""
        if self.current_letter != "None" and self.current_confidence > 50:
            self.sentence += self.current_letter
            self.update_sentence()

    def backspace(self, event):
        """Removes the last character from the sentence."""
        if self.sentence:
            self.sentence = self.sentence[:-1]
            self.update_sentence()

    def add_space(self, event):
        """Adds a space to the sentence."""
        self.sentence += " "
        self.update_sentence()

    def update_sentence(self):
        """Updates the sentence display in the GUI."""
        self.sentence_text.config(state='normal')
        self.sentence_text.delete(1.0, tk.END)
        self.sentence_text.insert(tk.END, self.sentence)
        self.sentence_text.config(state='disabled')

    def destroy(self):
        """Cleans up resources and closes the application."""
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.destroy)
    root.mainloop()
