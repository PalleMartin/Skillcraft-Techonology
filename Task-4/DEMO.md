# Hand Gesture Recognition System - Demo Guide

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Interface**:
   Open your browser and go to `http://localhost:5000`

## Using the System

### Option 1: Image Upload
1. Click on the "Choose File" button in the "Upload Image" section
2. Select an image containing a hand gesture
3. Click "Recognize Gesture"
4. View the predicted gesture and confidence score

### Option 2: Webcam Recognition
1. Click "Start Camera" to activate your webcam
2. Position your hand in front of the camera
3. Click "Capture & Recognize" to capture a frame and analyze it
4. View the predicted gesture and confidence score

## API Endpoints

### POST /predict
Upload an image file for gesture recognition:
```bash
curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
```

### POST /predict_frame
Send a base64-encoded image for recognition:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"image": "data:image/jpeg;base64,/9j/..."}' \
     http://localhost:5000/predict_frame
```

## Demo Mode

When TensorFlow is not available, the system runs in demo mode:
- All predictions return placeholder results
- Confidence scores are fixed values
- No actual model inference occurs

To enable full functionality, install TensorFlow:
```bash
pip install tensorflow
```

## Training Your Own Model

1. Prepare a dataset of hand gesture images
2. Organize images into class-specific folders
3. Modify `app/utils/model_trainer.py` with your dataset paths
4. Run the training script:
   ```bash
   python app/utils/model_trainer.py
   ```

## Supported Gestures

The current implementation supports 10 gesture classes:
1. Thumbs Up
2. Palm
3. Fist
4. Peace
5. Okay
6. Rock
7. Call
8. Love
9. Dislike
10. Like

## Troubleshooting

### Common Issues

1. **"Module not found" errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Camera not working**:
   - Check browser permissions for camera access
   - Ensure no other applications are using the camera

3. **Large file upload errors**:
   - The system limits uploads to 16MB
   - Compress images before uploading

4. **Slow prediction times**:
   - First prediction may be slow due to model loading
   - Subsequent predictions will be faster

## Extending the System

### Adding New Gestures
1. Add new class names to `GESTURE_CLASSES` in `app.py`
2. Retrain the model with the new gesture data
3. Update the frontend to display new gestures

### Customizing the UI
1. Modify `app/static/css/style.css` for visual changes
2. Update `app/templates/index.html` for layout changes
3. Enhance `app/static/js/script.js` for new interactions

## Performance Tips

1. **For Better Accuracy**:
   - Use high-quality images with clear hand visibility
   - Ensure good lighting conditions
   - Position hand in the center of the frame

2. **For Faster Processing**:
   - Resize images before uploading
   - Use JPEG format for smaller file sizes
   - Close other applications to free up system resources