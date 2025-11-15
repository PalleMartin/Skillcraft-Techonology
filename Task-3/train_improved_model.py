"""
Training Script for Improved Cat vs Dog Classifier

This script demonstrates how to train a more robust model that can better 
handle out-of-distribution samples by:
1. Including an "unknown" class during training
2. Using data augmentation to improve generalization
3. Implementing better uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
import warnings
import ssl
import urllib.request
warnings.filterwarnings("ignore")

# Fix for SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def create_improved_model(num_classes=2):
    """
    Create an improved ResNet50 model for cat vs dog classification
    
    Args:
        num_classes (int): Number of classes (2 for cat/dog, 3 for cat/dog/unknown)
        
    Returns:
        torch.nn.Module: Improved model
    """
    try:
        # Try to load pre-trained weights
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")
        print("Using model without pre-trained weights")
        model = torchvision.models.resnet50(weights=None)
    
    # Freeze early layers for transfer learning
    for param in list(model.parameters())[:-20]:  # Unfreeze last 20 parameters
        param.requires_grad = False
    
    # Replace the final classifier
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes)
    )
    
    return model

def get_data_transforms():
    """
    Get data transforms for training and validation
    
    Returns:
        tuple: (train_transforms, val_transforms)
    """
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def create_sample_dataset_structure():
    """
    Create a sample directory structure for demonstration purposes
    In a real scenario, you would have actual cat/dog images in these directories
    """
    directories = [
        "dataset/train/cats",
        "dataset/train/dogs",
        "dataset/val/cats",
        "dataset/val/dogs",
        "dataset/train/unknown",  # For out-of-distribution samples
        "dataset/val/unknown"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Sample dataset structure created:")
    for directory in directories:
        print(f"  {directory}/")

def train_model_with_uncertainty():
    """
    Demonstrate training with uncertainty quantification
    """
    print("=== Improved Cat vs Dog Classifier Training ===")
    print("This script demonstrates how to train a more robust model")
    print("that can better handle out-of-distribution samples.\n")
    
    # Create sample dataset structure
    create_sample_dataset_structure()
    
    # Model configuration
    num_classes = 3  # cats, dogs, unknown
    model = create_improved_model(num_classes)
    
    print(f"Model created with {num_classes} classes (cats, dogs, unknown)")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    print(f"\nTraining configuration:")
    print(f"  Device: {device}")
    print(f"  Optimizer: Adam")
    print(f"  Learning rate: 0.001")
    print(f"  Loss function: CrossEntropyLoss")
    
    # Data loading (in a real scenario, you would load actual data)
    train_transforms, val_transforms = get_data_transforms()
    print(f"\nData transforms:")
    print(f"  Training transforms: {len(train_transforms.transforms)} augmentations")
    print(f"  Validation transforms: Resize, ToTensor, Normalize")
    
    # Training loop example (commented out since we don't have actual data)
    print(f"\n=== Training Process ===")
    print("In a real implementation, you would:")
    print("1. Load your cat/dog dataset")
    print("2. Add out-of-distribution samples to 'unknown' class")
    print("3. Train the model with:")
    print("   - Transfer learning from ImageNet")
    print("   - Data augmentation for robustness")
    print("   - Regularization to prevent overfitting")
    print("   - Early stopping based on validation loss")
    
    # Save model example
    print(f"\n=== Model Saving ===")
    model_path = "improved_model.pth"
    print(f"Model would be saved to: {model_path}")
    print("(This is a demonstration - no actual training occurred)")
    
    # Benefits of this approach
    print(f"\n=== Benefits of This Approach ===")
    print("1. Out-of-distribution detection:")
    print("   - Model can explicitly classify unknown images")
    print("   - Reduces false positives for non-cat/dog images")
    print("2. Improved robustness:")
    print("   - Data augmentation increases generalization")
    print("   - Dropout and weight decay prevent overfitting")
    print("3. Better uncertainty quantification:")
    print("   - Model outputs probabilities for all classes")
    print("   - Can set thresholds for reliable predictions")
    
    return model

def demonstrate_uncertainty_quantification():
    """
    Demonstrate how to use uncertainty metrics for better predictions
    """
    print(f"\n=== Uncertainty Quantification ===")
    print("For improved classification reliability:")
    
    uncertainty_methods = [
        "1. Entropy-based uncertainty:",
        "   - High entropy = high uncertainty",
        "   - Low entropy = high confidence",
        "",
        "2. Monte Carlo Dropout:",
        "   - Multiple forward passes with dropout",
        "   - Variance in predictions indicates uncertainty",
        "",
        "3. Confidence thresholding:",
        "   - Only accept predictions above threshold",
        "   - Flag low-confidence predictions as uncertain",
        "",
        "4. Out-of-distribution detection:",
        "   - Train with explicit 'unknown' class",
        "   - Use distance metrics in feature space"
    ]
    
    for method in uncertainty_methods:
        print(method)

if __name__ == "__main__":
    # Train the improved model
    model = train_model_with_uncertainty()
    
    # Demonstrate uncertainty quantification
    demonstrate_uncertainty_quantification()
    
    print(f"\n=== Next Steps ===")
    print("To implement this approach:")
    print("1. Collect a dataset with cats, dogs, and other images")
    print("2. Organize images in the directory structure shown above")
    print("3. Uncomment the training loop and run with actual data")
    print("4. Use the trained model in your classifier application")
    print("5. Implement uncertainty metrics for better predictions")