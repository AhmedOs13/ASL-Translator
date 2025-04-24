# SilenTalker: Sign Language Recognition

SilenTalker is a real-time sign language recognition system that detects and classifies hand gestures for Arabic and English sign language alphabets using a webcam. It leverages deep learning models and MediaPipe for hand landmark detection, providing an intuitive GUI built with Tkinter.

## Features

- Supports both Arabic and English sign language alphabets.
- Real-time hand gesture recognition using a webcam.
- User-friendly GUI with model selection (Arabic/English).
- Displays predicted letters with confidence scores.
- Allows sentence construction with key bindings (Enter, Backspace, Space).
- Scalable and normalized hand landmark processing for robust detection.

## Prerequisites

- Python 3.8 or higher
- Webcam
- Pre-trained models: `arabic_sign_language_model.h5` and `english_sl_model_v2.h5` (not included in this repository due to size)

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/AhmedOs13/Sign-Language-Recognition.git
   cd Sign-Language-Recognition
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Place pre-trained models**:

   - The `arabic_sign_language_model.h5` and `english_sl_model_v2.h5` files are not included due to their size.
   - Contact the team or refer to model training documentation to obtain or train the models.
   - Place the `.h5` files in the root directory of the project.

## Usage

1. **Run the application**:

   ```bash
   python app.py
   ```

2. **Interact with the GUI**:

   - **Select Language Model**: Use the dropdown to switch between Arabic and English models.
   - **View Predictions**: The predicted letter and confidence score are displayed in real-time.
   - **Build Sentences**:
     - Press `Enter` to add the predicted letter to the sentence (if confidence &gt; 50%).
     - Press `Space` to add a space.
     - Press `Backspace` to remove the last character.
   - **Exit**: Close the window to stop the application.

3. **Ensure proper lighting and hand positioning**:

   - Keep your hand within the webcam's view.
   - Avoid complex backgrounds for better detection.

## Screenshots/Demo


## Project Structure

```
silentalker/
├── app.py                    # Main application script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore                # Git ignore file
├── arabic_sign_language_model.h5  # Arabic model (not included)
└── english_sl_model_v2.h5        # English model (not included)
```

## Dependencies

Listed in `requirements.txt`:

- opencv-python
- numpy
- mediapipe
- tensorflow
- pillow
- tkinter (usually included with Python)


## Limitations

- Requires pre-trained models, which are not included in this repository.
- Performance depends on lighting conditions and webcam quality.
- Only supports single-hand detection.
- Models may not generalize well to all hand sizes or backgrounds.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- **MediaPipe**: For providing robust hand landmark detection.
- **TensorFlow**: For enabling deep learning model development.
- Developed by the Biomedical Engineering team at Minia University.

For issues or feature requests, please open an issue on GitHub.
