"""
Advanced Cat vs Dog Classifier with Out-of-Distribution Detection

This script demonstrates a more robust approach to image classification that can
better detect when an image doesn't belong to either of the trained categories.
"""

import streamlit as st
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import gdown

# Download the model (only if not already downloaded)
model_path = "model_trained.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1_i4KD6zChAqoh94FeBSJHSoRvT4XsscU"
    gdown.download(url, model_path, quiet=False)

# Title and subtitle
st.set_page_config(page_title="Advanced Cat vs Dog Classifier 🐱🐶", page_icon="🐾")
st.markdown(
    """
    <h1 style='text-align: center; color: #6C3483;'>🐾 Advanced Cat vs Dog Classifier 🐾</h1>
    <p style='text-align: center; font-size:20px;'>Upload an image and see what our model predicts!</p>
    """,
    unsafe_allow_html=True,
)

# Load the model
@st.cache_resource
def load_model():
    """Load the pre-trained ResNet50 model for cat vs dog classification"""
    model = torchvision.models.resnet50(weights=None)
    classifier = torch.nn.Sequential(
        torch.nn.Linear(2048, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 2)
    )
    model.fc = classifier
    model.load_state_dict(torch.load("model_trained.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

@st.cache_resource
def get_model():
    """Get the loaded model with caching"""
    return load_model()

def calculate_prediction_metrics(probabilities):
    """
    Calculate various metrics to assess prediction reliability
    
    Args:
        probabilities: softmax probabilities from the model
        
    Returns:
        dict: Dictionary containing various uncertainty metrics
    """
    # Convert to numpy if tensor
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.cpu().numpy().squeeze()
    else:
        probs = np.array(probabilities)
    
    # Basic metrics
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)
    
    # Entropy calculation (measure of uncertainty)
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    clipped_probs = np.clip(probs, eps, 1 - eps)
    entropy = -np.sum(clipped_probs * np.log(clipped_probs))
    
    # For binary classification, max entropy is log(2) ≈ 0.693
    normalized_entropy = entropy / np.log(2)
    
    # Margin between top two predictions
    sorted_probs = np.sort(probs)[::-1]  # Descending order
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    
    return {
        'max_probability': max_prob,
        'predicted_class': predicted_class,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'margin': margin,
        'probabilities': probs
    }

def classify_image_with_uncertainty(image, model):
    """
    Classify an image and provide uncertainty estimates
    
    Args:
        image: PIL Image object
        model: PyTorch model
        
    Returns:
        dict: Classification results with uncertainty metrics
    """
    # Preprocess image
    img_tensor = transform(image)
    img_batch = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(img_batch)
        probabilities = F.softmax(outputs, dim=1)
    
    # Calculate uncertainty metrics
    metrics = calculate_prediction_metrics(probabilities)
    
    return metrics

def interpret_prediction(metrics):
    """
    Interpret the prediction based on uncertainty metrics
    
    Args:
        metrics: Dictionary of prediction metrics
        
    Returns:
        dict: Interpretation results
    """
    max_prob = metrics['max_probability']
    entropy = metrics['normalized_entropy']
    margin = metrics['margin']
    
    # Multi-criteria decision making for reliability
    # 1. High confidence (high max probability)
    is_high_confidence = max_prob > 0.7
    
    # 2. Low uncertainty (low entropy)
    is_low_uncertainty = entropy < 0.7  # Less than 70% of maximum entropy
    
    # 3. Clear decision boundary (large margin)
    is_clear_margin = margin > 0.2
    
    # Count how many criteria are met
    criteria_met = sum([is_high_confidence, is_low_uncertainty, is_clear_margin])
    
    # Determine overall reliability
    if criteria_met >= 2:
        reliability = "High"
        interpretation = "The model is confident in its prediction."
        category = "reliable"
    elif criteria_met == 1:
        reliability = "Medium"
        interpretation = "The model is somewhat uncertain about this prediction."
        category = "uncertain"
    else:
        reliability = "Low"
        interpretation = "The model is highly uncertain. This image may not be a cat or dog."
        category = "unreliable"
    
    return {
        'reliability': reliability,
        'interpretation': interpretation,
        'category': category,
        'criteria_met': criteria_met
    }

# Main application
model = get_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image 🖼️", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Classify the image
    metrics = classify_image_with_uncertainty(image, model)
    interpretation = interpret_prediction(metrics)
    
    class_names = ["Cat", "Dog"]
    predicted_label = class_names[metrics['predicted_class']]
    confidence_percent = metrics['max_probability'] * 100
    
    # Display results based on reliability
    if interpretation['category'] == 'reliable':
        st.markdown(
            f"<h2 style='text-align: center; color: #1F618D;'>Prediction: {predicted_label} 🐾</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align: center;'>Confidence: {confidence_percent:.2f}%</h3>",
            unsafe_allow_html=True
        )
    elif interpretation['category'] == 'uncertain':
        st.markdown(
            f"<h2 style='text-align: center; color: #FF8C00;'>⚠️ Uncertain Prediction</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align: center;'>Most likely: {predicted_label} ({confidence_percent:.2f}%)</h3>",
            unsafe_allow_html=True
        )
        st.info(interpretation['interpretation'])
    else:  # unreliable
        st.markdown(
            f"<h2 style='text-align: center; color: #E74C3C;'>❌ Not a Cat or Dog</h2>",
            unsafe_allow_html=True
        )
        st.info(interpretation['interpretation'])
    
    # Detailed metrics display
    st.subheader("Prediction Analysis 📊")
    
    # Show probabilities
    st.write("**Class Probabilities:**")
    prob_chart_data = {
        "Cat": [metrics['probabilities'][0]],
        "Dog": [metrics['probabilities'][1]]
    }
    st.bar_chart(prob_chart_data)
    
    # Show uncertainty metrics
    st.write("**Uncertainty Metrics:**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Confidence", f"{confidence_percent:.1f}%")
    col2.metric("Normalized Entropy", f"{metrics['normalized_entropy']:.3f}")
    col3.metric("Decision Margin", f"{metrics['margin']:.3f}")
    
    # Show reliability assessment
    st.write("**Reliability Assessment:**")
    reliability_score = interpretation['criteria_met'] / 3.0
    reliability_desc = interpretation['reliability']
    st.progress(reliability_score, text=f"Reliability: {reliability_desc}")
    
    # Additional information
    if interpretation['category'] == 'unreliable':
        st.warning("""
        **Why this might be happening:**
        - The image is not a cat or dog
        - The image quality is poor
        - The subject is in an unusual pose or setting
        - The image contains multiple objects
        
        **Recommendation:** Try uploading a clear photo of a cat or dog for better results.
        """)

else:
    st.info("👈 Upload an image to start!", icon="ℹ️")

# Add information about the approach
with st.expander("ℹ️ How This Advanced Classifier Works"):
    st.markdown("""
    This advanced classifier uses multiple techniques to assess prediction reliability:
    
    **Uncertainty Metrics:**
    1. **Confidence**: How certain the model is about its top prediction
    2. **Entropy**: Measures the "randomness" of predictions (lower = more certain)
    3. **Margin**: Difference between the top two predictions (larger = more certain)
    
    **Reliability Assessment:**
    - **High Reliability**: At least 2 of 3 criteria indicate confidence
    - **Medium Reliability**: Only 1 criterion indicates confidence
    - **Low Reliability**: None or only 1 criterion indicates confidence
    
    **Benefits:**
    - Better detection of non-cat/dog images
    - More informative feedback when predictions are uncertain
    - Reduced false positives for out-of-distribution images
    """)