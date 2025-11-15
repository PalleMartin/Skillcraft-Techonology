import streamlit as st
import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import gdown
import torch

# Download the model (only if not already downloaded)
model_path = "model_trained.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1_i4KD6zChAqoh94FeBSJHSoRvT4XsscU"
    gdown.download(url, model_path, quiet=False)



# Title and subtitle
st.set_page_config(page_title="Cat vs Dog Classifier 🐱🐶", page_icon="🐾")
st.markdown(
    """
    <h1 style='text-align: center; color: #6C3483;'>🐾 Cat vs Dog Classifier 🐾</h1>
    <p style='text-align: center; font-size:20px;'>Upload an image and see what our model predicts!</p>
    <p style='text-align: center; font-size:16px; color: #888888;'>Note: This model is trained specifically to distinguish between cats and dogs. Other images may produce inaccurate predictions.</p>
    """,
    unsafe_allow_html=True,
)

# Load the model
@st.cache_resource
def load_model():
    model = torchvision.models.resnet50(weights=None)  # No need to download weights
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

model = load_model()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# File uploader
uploaded_file = st.file_uploader("Upload a cat or dog image 🖼️", type=["jpg", "jpeg", "png"])

# If user uploads a file
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image)
    img_batch = img_tensor.unsqueeze(0)  # Add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(img_batch)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_names = ["Cat", "Dog"]
    prediction_label = class_names[predicted.item()]
    prediction_confidence = confidence.item() * 100
    
    # Set a confidence threshold for reliable predictions
    confidence_threshold = 70.0

    # Display prediction
    if prediction_confidence >= confidence_threshold:
        st.markdown(
            f"<h2 style='text-align: center; color: #1F618D;'>Prediction: {prediction_label} 🐾</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align: center;'>Confidence: {prediction_confidence:.2f}%</h3>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<h2 style='text-align: center; color: #FF8C00;'>⚠️ Low Confidence Prediction</h2>",
            unsafe_allow_html=True
        )
        st.markdown(
            f"<h3 style='text-align: center;'>Most likely: {prediction_label} ({prediction_confidence:.2f}%)</h3>",
            unsafe_allow_html=True
        )
        st.info("💡 Note: The model is not confident in this prediction. This image might not be a cat or dog.")

    # Display probability bar chart
    st.subheader("Prediction Probabilities 📊")
    prob_array = probabilities.squeeze().cpu().numpy()
    st.bar_chart({
        "Cat": [prob_array[0]],
        "Dog": [prob_array[1]]
    })
    
    # Additional information about confidence
    if prediction_confidence < confidence_threshold:
        st.warning("The model has low confidence in its prediction. This could mean:\n"
                  "- The image is neither a cat nor a dog\n"
                  "- The image quality is poor\n"
                  "- The image is a cat or dog but in an unusual pose or setting")

else:
    st.info("👈 Upload an image to start!", icon="ℹ️")
    
# Add information about the model
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    This classifier was trained on a dataset of cat and dog images. It can only reliably distinguish between these two categories.
    
    **Tips for best results:**
    - Use clear photos of cats or dogs
    - Ensure the pet occupies a significant portion of the image
    - Avoid group photos or images with multiple animals
    - Images of other subjects (humans, plants, objects) will likely produce inaccurate results
    
    **Model Details:**
    - Architecture: ResNet-50
    - Categories: Cats vs Dogs
    """)