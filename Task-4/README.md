# TASK 04 – Hand Gesture Recognition System

## Overview
This is a Python Full-Stack project called 'Hand Gesture Recognition System' that uses deep learning and computer vision to accurately identify and classify hand gestures from image or video input. The backend is built using Flask and integrates a trained CNN model (TensorFlow/Keras) for prediction, while the frontend provides a user interface using HTML, CSS and JavaScript to upload images or enable webcam recognition.

## Features
- Real-time gesture detection from webcam feed
- Image upload functionality for gesture recognition
- Modern tech-themed UI with responsive design
- Intuitive human-computer interaction
- Seamless gesture-based control

## Technologies Used
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras (CNN Model)
- **Computer Vision**: OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Deployment**: Local server

## Project Structure
```
Task-4/
│
├── app/
│   ├── templates/          # HTML templates
│   ├── static/             # CSS, JS, and other static files
│   │   ├── css/
│   │   └── js/
│   ├── models/             # Trained ML models
│   └── utils/              # Utility scripts
│
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Task-4
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```bash
   python app.py
   ```

4. Open your browser and go to `http://localhost:5000`

## Usage

1. **Image Upload**: Click on the "Choose File" button to upload an image containing a hand gesture, then click "Recognize Gesture".

2. **Webcam Recognition**: Click "Start Camera" to activate your webcam, then "Capture & Recognize" to capture a frame and recognize the gesture.

## Model Training

To train the gesture recognition model:

1. Prepare your dataset of hand gesture images
2. Modify the `app/utils/model_trainer.py` script with your dataset paths
3. Run the training script:
   ```bash
   python app/utils/model_trainer.py
   ```

## Gesture Classes
The current implementation supports the following gesture classes:
- Thumbs Up
- Palm
- Fist
- Peace
- Okay
- Rock
- Call
- Love
- Dislike
- Like

## Future Improvements
- Implement transfer learning with pre-trained models for better accuracy
- Add support for more gesture classes
- Improve real-time performance
- Add gesture history tracking
- Implement gesture-based command execution

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.