"""
Test script to verify that required packages are installed
"""

def test_imports():
    """Test if required packages can be imported"""
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "torch":
                import torch
                print(f"✓ {package} version: {torch.__version__}")
            elif package == "torchvision":
                import torchvision
                print(f"✓ {package} version: {torchvision.__version__}")
            elif package == "numpy":
                import numpy as np
                print(f"✓ {package} version: {np.__version__}")
            elif package == "matplotlib":
                import matplotlib
                print(f"✓ {package} version: {matplotlib.__version__}")
        except ImportError as e:
            print(f"✗ {package} not found: {e}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install torch torchvision numpy matplotlib")
        return False
    else:
        print("\nAll required packages are available!")
        return True

def test_dataset_structure():
    """Test if dataset structure is correct"""
    import os
    
    required_dirs = [
        "dataset/train/cats",
        "dataset/train/dogs",
        "dataset/train/unknown",
        "dataset/val/cats",
        "dataset/val/dogs",
        "dataset/val/unknown"
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            # Count files in directory
            files = os.listdir(directory)
            print(f"✓ {directory}: {len(files)} files")
        else:
            print(f"✗ {directory}: Not found")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\nMissing directories: {', '.join(missing_dirs)}")
        print("Please run organize_dataset.py to create the dataset structure")
        return False
    else:
        print("\nDataset structure is correct!")
        return True

def main():
    """Main test function"""
    print("=== Environment Test ===")
    print()
    
    imports_ok = test_imports()
    print()
    
    dataset_ok = test_dataset_structure()
    print()
    
    if imports_ok and dataset_ok:
        print("🎉 Environment is ready for training!")
        print("\nTo train the model, run:")
        print("python train_with_organized_dataset.py")
    else:
        print("❌ Environment is not ready for training")
        if not imports_ok:
            print("  - Missing required packages")
        if not dataset_ok:
            print("  - Dataset structure issues")

if __name__ == "__main__":
    main()