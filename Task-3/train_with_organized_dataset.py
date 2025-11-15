"""
Training Script Using Organized Dataset

This script demonstrates how to train an improved cat vs dog classifier 
using the properly organized dataset with cat, dog, and unknown images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import ssl

# Fix for SSL certificate issue on macOS
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def create_improved_model(num_classes=3):
    """
    Create an improved ResNet50 model for cat vs dog vs unknown classification
    
    Args:
        num_classes (int): Number of classes (3 for cat/dog/unknown)
        
    Returns:
        torch.nn.Module: Improved model
    """
    try:
        # Try to load pre-trained weights
        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        print("Loaded pre-trained ResNet50 weights")
    except Exception as e:
        print(f"Warning: Could not load pre-trained weights: {e}")
        print("Using model without pre-trained weights")
        model = torchvision.models.resnet50(weights=None)
    
    # Freeze early layers for transfer learning (unfreeze last 20 parameters)
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Replace the final classifier for 3 classes
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

def load_dataset():
    """
    Load the organized dataset
    
    Returns:
        tuple: (train_dataset, val_dataset, class_names)
    """
    # Define data transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root="dataset/train",
        transform=train_transforms
    )
    
    val_dataset = datasets.ImageFolder(
        root="dataset/val",
        transform=val_transforms
    )
    
    class_names = train_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Count samples per class
    train_counts = {}
    for _, label in train_dataset:
        class_name = class_names[label]
        train_counts[class_name] = train_counts.get(class_name, 0) + 1
    
    val_counts = {}
    for _, label in val_dataset:
        class_name = class_names[label]
        val_counts[class_name] = val_counts.get(class_name, 0) + 1
    
    print("\nTraining set distribution:")
    for class_name, count in train_counts.items():
        print(f"  {class_name}: {count} images")
    
    print("\nValidation set distribution:")
    for class_name, count in val_counts.items():
        print(f"  {class_name}: {count} images")
    
    return train_dataset, val_dataset, class_names

def train_model(model, train_loader, val_loader, class_names, num_epochs=10):
    """
    Train the model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_names: List of class names
        num_epochs: Number of training epochs
        
    Returns:
        trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model saved with validation accuracy: {val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)
    
    return model

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracies
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("Training curves saved as 'training_curves.png'")
    plt.show()

def main():
    """
    Main training function
    """
    print("=== Enhanced Cat vs Dog Classifier Training ===")
    print("Using organized dataset with cat, dog, and unknown images")
    print()
    
    # Check if dataset exists
    if not os.path.exists("dataset/train") or not os.path.exists("dataset/val"):
        print("Error: Dataset not found!")
        print("Please run organize_dataset.py first to create the dataset structure.")
        return
    
    # Load dataset
    try:
        train_dataset, val_dataset, class_names = load_dataset()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Create model
    print("\nCreating improved model...")
    model = create_improved_model(num_classes=len(class_names))
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model (commented out for demonstration)
    print("\nTo train the model, uncomment the following lines:")
    print("# trained_model = train_model(model, train_loader, val_loader, class_names, num_epochs=10)")
    print("# torch.save(trained_model.state_dict(), 'final_model.pth')")
    print("# print('Model saved as final_model.pth')")
    
    # In a real scenario, you would uncomment the lines above to actually train
    print("\nNote: This is a demonstration script.")
    print("In a real implementation, you would:")
    print("1. Uncomment the training code above")
    print("2. Adjust hyperparameters as needed")
    print("3. Use a GPU if available for faster training")
    print("4. Monitor training with TensorBoard or similar tools")

if __name__ == "__main__":
    main()