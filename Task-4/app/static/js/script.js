document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const uploadForm = document.getElementById('upload-form');
    const imageInput = document.getElementById('image-input');
    const resultDiv = document.getElementById('result');
    const startCameraBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-btn');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const webcamResult = document.getElementById('webcam-result');
    const loading = document.getElementById('loading');
    
    let stream = null;
    
    // Handle image upload and prediction
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!imageInput.files.length) {
            showResult(resultDiv, 'Please select an image', 'error');
            return;
        }
        
        const file = imageInput.files[0];
        const formData = new FormData();
        formData.append('image', file);
        
        showLoading(true);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            showLoading(false);
            
            if (data.success) {
                showResult(resultDiv, `Gesture: ${data.gesture}<br>Confidence: ${(data.confidence * 100).toFixed(2)}%`, 'success');
            } else {
                showResult(resultDiv, `Error: ${data.error}`, 'error');
            }
        } catch (error) {
            showLoading(false);
            showResult(resultDiv, `Error: ${error.message}`, 'error');
        }
    });
    
    // Start camera for webcam recognition
    startCameraBtn.addEventListener('click', async function() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            captureBtn.disabled = false;
        } catch (error) {
            showResult(webcamResult, `Error accessing camera: ${error.message}`, 'error');
        }
    });
    
    // Capture frame and send for prediction
    captureBtn.addEventListener('click', async function() {
        if (!stream) return;
        
        // Draw current video frame to canvas
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Convert canvas to blob and then to base64
        canvas.toBlob(async function(blob) {
            const reader = new FileReader();
            reader.onload = async function() {
                const imageData = reader.result;
                
                showLoading(true);
                
                try {
                    const response = await fetch('/predict_frame', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ image: imageData })
                    });
                    
                    const data = await response.json();
                    showLoading(false);
                    
                    if (data.success) {
                        showResult(webcamResult, `Gesture: ${data.gesture}<br>Confidence: ${(data.confidence * 100).toFixed(2)}%`, 'success');
                    } else {
                        showResult(webcamResult, `Error: ${data.error}`, 'error');
                    }
                } catch (error) {
                    showLoading(false);
                    showResult(webcamResult, `Error: ${error.message}`, 'error');
                }
            };
            reader.readAsDataURL(blob);
        }, 'image/jpeg');
    });
    
    // Helper functions
    function showResult(element, message, type) {
        element.innerHTML = `<div class="gesture-display">${message}</div>`;
        element.className = type === 'success' ? 'result-success' : 'result-error';
    }
    
    function showLoading(show) {
        if (show) {
            loading.classList.remove('hidden');
        } else {
            loading.classList.add('hidden');
        }
    }
});