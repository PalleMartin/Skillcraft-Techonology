"""
Dataset Organization Script

This script helps organize cat, dog, and unknown images into the proper directory structure
for training an improved cat vs dog classifier.
"""

import os
import shutil
from PIL import Image
import numpy as np

def create_sample_images():
    """
    Create sample images for demonstration purposes
    In a real scenario, you would download or collect actual images
    """
    # Create directories for sample images
    sample_dirs = [
        "sample_images/cats",
        "sample_images/dogs",
        "sample_images/unknown"
    ]
    
    for directory in sample_dirs:
        os.makedirs(directory, exist_ok=True)
    
    # Create sample images (simple colored squares for demonstration)
    def create_sample_image(color, path):
        """Create a simple colored square image"""
        img = Image.new('RGB', (224, 224), color)
        img.save(path)
    
    # Create sample cat images (orange)
    for i in range(5):
        create_sample_image((255, 165, 0), f"sample_images/cats/cat_sample_{i}.jpg")
    
    # Create sample dog images (brown)
    for i in range(5):
        create_sample_image((139, 69, 19), f"sample_images/dogs/dog_sample_{i}.jpg")
    
    # Create sample unknown images (gray)
    for i in range(5):
        create_sample_image((128, 128, 128), f"sample_images/unknown/unknown_sample_{i}.jpg")
    
    print("Sample images created:")
    print(f"  - {len(os.listdir('sample_images/cats'))} cat samples")
    print(f"  - {len(os.listdir('sample_images/dogs'))} dog samples")
    print(f"  - {len(os.listdir('sample_images/unknown'))} unknown samples")

def organize_dataset_structure():
    """
    Organize the dataset structure for training
    """
    print("=== Dataset Organization Script ===")
    print("This script demonstrates how to organize your dataset for training an improved classifier.")
    print()
    
    # Create the proper directory structure if it doesn't exist
    directories = [
        "dataset/train/cats",
        "dataset/train/dogs",
        "dataset/train/unknown",
        "dataset/val/cats",
        "dataset/val/dogs",
        "dataset/val/unknown"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")
    
    print()
    print("Dataset structure is ready!")
    print("Place your actual images in the following directories:")
    print("  - dataset/train/cats/      (training cat images)")
    print("  - dataset/train/dogs/      (training dog images)")
    print("  - dataset/train/unknown/   (training unknown images)")
    print("  - dataset/val/cats/        (validation cat images)")
    print("  - dataset/val/dogs/        (validation dog images)")
    print("  - dataset/val/unknown/     (validation unknown images)")

def copy_sample_images_to_dataset():
    """
    Copy sample images to demonstrate the dataset organization
    """
    # Create sample images first
    create_sample_images()
    
    # Copy sample images to training directories
    print("Copying sample images to dataset structure...")
    
    # Copy cat samples
    cat_samples = os.listdir("sample_images/cats")
    for i, sample in enumerate(cat_samples):
        src = os.path.join("sample_images/cats", sample)
        # Split between train and val (80/20 split)
        if i < len(cat_samples) * 0.8:
            dst = os.path.join("dataset/train/cats", sample)
        else:
            dst = os.path.join("dataset/val/cats", sample)
        shutil.copy(src, dst)
    
    # Copy dog samples
    dog_samples = os.listdir("sample_images/dogs")
    for i, sample in enumerate(dog_samples):
        src = os.path.join("sample_images/dogs", sample)
        # Split between train and val (80/20 split)
        if i < len(dog_samples) * 0.8:
            dst = os.path.join("dataset/train/dogs", sample)
        else:
            dst = os.path.join("dataset/val/dogs", sample)
        shutil.copy(src, dst)
    
    # Copy unknown samples
    unknown_samples = os.listdir("sample_images/unknown")
    for i, sample in enumerate(unknown_samples):
        src = os.path.join("sample_images/unknown", sample)
        # Split between train and val (80/20 split)
        if i < len(unknown_samples) * 0.8:
            dst = os.path.join("dataset/train/unknown", sample)
        else:
            dst = os.path.join("dataset/val/unknown", sample)
        shutil.copy(src, dst)
    
    print("Sample images copied to dataset directories:")
    print(f"  - Train cats: {len(os.listdir('dataset/train/cats'))} images")
    print(f"  - Train dogs: {len(os.listdir('dataset/train/dogs'))} images")
    print(f"  - Train unknown: {len(os.listdir('dataset/train/unknown'))} images")
    print(f"  - Val cats: {len(os.listdir('dataset/val/cats'))} images")
    print(f"  - Val dogs: {len(os.listdir('dataset/val/dogs'))} images")
    print(f"  - Val unknown: {len(os.listdir('dataset/val/unknown'))} images")

def show_dataset_statistics():
    """
    Show statistics about the current dataset
    """
    print("\n=== Dataset Statistics ===")
    
    try:
        train_cats = len(os.listdir("dataset/train/cats"))
        train_dogs = len(os.listdir("dataset/train/dogs"))
        train_unknown = len(os.listdir("dataset/train/unknown"))
        val_cats = len(os.listdir("dataset/val/cats"))
        val_dogs = len(os.listdir("dataset/val/dogs"))
        val_unknown = len(os.listdir("dataset/val/unknown"))
        
        print("Training set:")
        print(f"  - Cats: {train_cats} images")
        print(f"  - Dogs: {train_dogs} images")
        print(f"  - Unknown: {train_unknown} images")
        print(f"  - Total: {train_cats + train_dogs + train_unknown} images")
        
        print("Validation set:")
        print(f"  - Cats: {val_cats} images")
        print(f"  - Dogs: {val_dogs} images")
        print(f"  - Unknown: {val_unknown} images")
        print(f"  - Total: {val_cats + val_dogs + val_unknown} images")
        
        print("Overall:")
        print(f"  - Total images: {train_cats + train_dogs + train_unknown + val_cats + val_dogs + val_unknown}")
        print(f"  - Classes: 3 (cats, dogs, unknown)")
        
    except FileNotFoundError as e:
        print(f"Dataset directories not found: {e}")
        print("Please run the organize_dataset_structure() function first.")

def main():
    """
    Main function to organize the dataset
    """
    print("Cat vs Dog Dataset Organizer")
    print("============================")
    
    # Show current dataset status
    show_dataset_statistics()
    
    print("\nOptions:")
    print("1. Create proper dataset directory structure")
    print("2. Copy sample images to demonstrate organization")
    print("3. Show dataset statistics")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                organize_dataset_structure()
            elif choice == "2":
                copy_sample_images_to_dataset()
            elif choice == "3":
                show_dataset_statistics()
            elif choice == "4":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()