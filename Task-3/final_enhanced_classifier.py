"""
Final Enhanced Cat vs Dog Classifier

This is the final enhanced version that combines all improvements:
1. Better uncertainty detection using multiple metrics
2. Clearer feedback when images are not cats or dogs
3. More informative UI with detailed analysis
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
    try:
        gdown.download(url, model_path, quiet=False)
    except Exception as e:
        st.error(f"Could not download model: {e}")
        st.info("Please check your internet connection and try again.")

# Title and subtitle
st.set_page_config(page_title="Enhanced Cat vs Dog Classifier 🐱🐶", page_icon="🐾")
st.markdown(
    """
    <h1 style='text-align: center; color: #6C3483;'>🐾 Enhanced Cat vs Dog Classifier 🐾</h1>
    <p style='text-align: center; font-size:20px;'>Upload an image and see what our improved model predicts!</p>
    """,
    unsafe_allow_html=True,
)

# Load the model
@st.cache_resource
def load_model():
    """Load the pre-trained ResNet50 model for cat vs dog classification"""
    try:
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
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

def calculate_entropy(probabilities):
    """Calculate entropy of prediction probabilities"""
    # Convert to numpy if needed
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.cpu().numpy().squeeze()
    else:
        probs = np.array(probabilities)
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-8
    probs = np.clip(probs, eps, 1 - eps)
    
    # Calculate entropy: -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def analyze_prediction_confidence(probabilities):
    """
    Analyze prediction confidence using multiple metrics
    
    Args:
        probabilities: softmax probabilities from the model
        
    Returns:
        dict: Analysis results including confidence metrics
    """
    # Convert to numpy if needed
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.cpu().numpy().squeeze()
    else:
        probs = np.array(probabilities)
    
    # Basic metrics
    max_prob = np.max(probs)
    predicted_class = np.argmax(probs)
    
    # Entropy calculation (measure of uncertainty)
    entropy = calculate_entropy(probs)
    # Normalize entropy by max possible entropy for 2 classes (log(2))
    normalized_entropy = entropy / np.log(2)
    
    # Margin between top two predictions
    sorted_probs = np.sort(probs)[::-1]  # Descending order
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0
    
    # Multi-criteria decision making
    # Criterion 1: High confidence (high max probability)
    is_high_confidence = max_prob > 0.7
    
    # Criterion 2: Low uncertainty (low entropy)
    is_low_entropy = normalized_entropy < 0.7  # Less than 70% of maximum entropy
    
    # Criterion 3: Clear decision boundary (large margin)
    is_clear_margin = margin > 0.2
    
    # Count how many criteria are met
    criteria_met = sum([is_high_confidence, is_low_entropy, is_clear_margin])
    
    # Determine overall reliability
    if criteria_met >= 2:
        reliability = "High"
        confidence_level = "reliable"
    elif criteria_met == 1:
        reliability = "Medium"
        confidence_level = "uncertain"
    else:
        reliability = "Low"
        confidence_level = "unreliable"
    
    return {
        'max_probability': float(max_prob),
        'predicted_class': int(predicted_class),
        'entropy': float(entropy),
        'normalized_entropy': float(normalized_entropy),
        'margin': float(margin),
        'probabilities': probs.tolist(),
        'reliability': reliability,
        'confidence_level': confidence_level,
        'criteria_met': criteria_met,
        'is_high_confidence': is_high_confidence,
        'is_low_entropy': is_low_entropy,
        'is_clear_margin': is_clear_margin
    }

def classify_image(image, model):
    """
    Classify an image using the model
    
    Args:
        image: PIL Image object
        model: PyTorch model
        
    Returns:
        dict: Classification results with analysis
    """
    try:
        # Preprocess image
        img_tensor = transform(image)
        img_batch = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(img_batch)
            probabilities = F.softmax(outputs, dim=1)
        
        # Analyze confidence
        analysis = analyze_prediction_confidence(probabilities)
        
        return analysis
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# Main application
model = get_model()

if model is None:
    st.error("Failed to load model. Please check the model file.")
else:
    # File uploader
    uploaded_file = st.file_uploader("Upload an image 🖼️", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Classify the image
            analysis = classify_image(image, model)
            
            if analysis is not None:
                class_names = ["Cat", "Dog"]
                predicted_label = class_names[analysis['predicted_class']]
                confidence_percent = analysis['max_probability'] * 100
                
                # Display results based on confidence level
                if analysis['confidence_level'] == 'reliable':
                    st.markdown(
                        f"<h2 style='text-align: center; color: #1F618D;'>Prediction: {predicted_label} 🐾</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<h3 style='text-align: center;'>Confidence: {confidence_percent:.2f}%</h3>",
                        unsafe_allow_html=True
                    )
                elif analysis['confidence_level'] == 'uncertain':
                    st.markdown(
                        f"<h2 style='text-align: center; color: #FF8C00;'>⚠️ Uncertain Prediction</h2>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<h3 style='text-align: center;'>Most likely: {predicted_label} ({confidence_percent:.2f}%)</h3>",
                        unsafe_allow_html=True
                    )
                    st.info("💡 The model is somewhat uncertain about this prediction.")
                else:  # unreliable
                    st.markdown(
                        f"<h2 style='text-align: center; color: #E74C3C;'>❌ Likely Not a Cat or Dog</h2>",
                        unsafe_allow_html=True
                    )
                    st.info("💡 This image doesn't appear to be a cat or dog. The model is highly uncertain about this prediction.")
                
                # Detailed metrics display
                st.subheader("Prediction Analysis 📊")
                
                # Show probabilities
                st.write("**Class Probabilities:**")
                prob_chart_data = {
                    "Cat": [analysis['probabilities'][0]],
                    "Dog": [analysis['probabilities'][1]]
                }
                st.bar_chart(prob_chart_data)
                
                # Show uncertainty metrics
                st.write("**Uncertainty Metrics:**")
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", f"{confidence_percent:.1f}%")
                col2.metric("Normalized Entropy", f"{analysis['normalized_entropy']:.3f}")
                col3.metric("Decision Margin", f"{analysis['margin']:.3f}")
                
                # Show reliability assessment
                st.write("**Reliability Assessment:**")
                reliability_score = analysis['criteria_met'] / 3.0
                reliability_desc = analysis['reliability']
                st.progress(reliability_score, text=f"Reliability: {reliability_desc}")
                
                # Additional information for low reliability
                if analysis['confidence_level'] == 'unreliable':
                    st.warning("""
                    **Why this might be happening:**
                    - The image is not a cat or dog
                    - The image quality is poor
                    - The subject is in an unusual pose or setting
                    - The image contains multiple objects
                    
                    **Recommendation:** Try uploading a clear photo of a cat or dog for better results.
                    """)
                
                # Show detailed criteria
                with st.expander("🔍 Detailed Criteria Analysis"):
                    st.write("**Confidence Criteria:**")
                    st.write(f"✓ High Confidence (>70%): {'Yes' if analysis['is_high_confidence'] else 'No'}")
                    st.write(f"✓ Low Entropy (<70% max): {'Yes' if analysis['is_low_entropy'] else 'No'}")
                    st.write(f"✓ Clear Margin (>0.2): {'Yes' if analysis['is_clear_margin'] else 'No'}")
                    st.write(f"**Criteria Met: {analysis['criteria_met']}/3**")
                    
            else:
                st.error("Failed to analyze the image. Please try another image.")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        st.info("👈 Upload an image to start!", icon="ℹ️")

# Add information about the approach
with st.expander("ℹ️ How This Enhanced Classifier Works"):
    st.markdown("""
    This enhanced classifier uses multiple techniques to better assess prediction reliability:
    
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
    
    st.markdown("""
    **For AIML Students:**
    To further improve this classifier, consider:
    1. Training with an explicit "unknown" class
    2. Using Monte Carlo Dropout for better uncertainty estimation
    3. Implementing temperature scaling for better calibrated probabilities
    4. Adding more diverse out-of-distribution samples during training
    """)