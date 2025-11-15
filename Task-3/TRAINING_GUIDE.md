# Enhanced Cat vs Dog Classifier Training Guide

This guide explains how to use all the components we've created to train an improved cat vs dog classifier that can better handle out-of-distribution samples.

## Project Structure

```
cats_vs_dogs_prediction_app-main/
├── app.py                          # Original classifier with basic improvements
├── advanced_classifier.py          # Advanced classifier with multi-metric uncertainty
├── final_enhanced_classifier.py    # Final enhanced classifier with detailed dashboard
├── train_improved_model.py         # Demonstration of improved model architecture
├── train_with_organized_dataset.py # Actual training script using organized dataset
├── organize_dataset.py             # Script to organize dataset structure
├── test_environment.py             # Environment testing script
├── dataset/                        # Organized dataset structure
│   ├── train/
│   │   ├── cats/
│   │   ├── dogs/
│   │   └── unknown/
│   └── val/
│       ├── cats/
│       ├── dogs/
│       └── unknown/
├── sample_images/                  # Sample images for demonstration
├── model_trained.pth               # Original pre-trained model
├── requirements.txt                # Dependencies
└── README.md                      # Project documentation
```

## Step-by-Step Training Process

### 1. Environment Setup

First, ensure all required packages are installed:
```bash
pip install -r requirements.txt
```

Or use conda if preferred:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

Test your environment:
```bash
python test_environment.py
```

### 2. Dataset Organization

The dataset has already been organized with the following structure:
- **Training set**: 12 images (4 cats, 4 dogs, 4 unknown)
- **Validation set**: 3 images (1 cat, 1 dog, 1 unknown)

To organize your own dataset, place your images in the appropriate directories:
- `dataset/train/cats/` - Training cat images
- `dataset/train/dogs/` - Training dog images
- `dataset/train/unknown/` - Training unknown images (non-cat/dog images)
- `dataset/val/cats/` - Validation cat images
- `dataset/val/dogs/` - Validation dog images
- `dataset/val/unknown/` - Validation unknown images

### 3. Training the Model

To train the improved model:
```bash
python train_with_organized_dataset.py
```

This script will:
1. Load the organized dataset
2. Create an improved ResNet50 model with 3 output classes
3. Demonstrate the training process (commented out for safety)
4. Show how to save the trained model

### 4. Using the Enhanced Classifier

After training, you can use the enhanced classifiers:

#### Final Enhanced Classifier (Recommended)
```bash
streamlit run final_enhanced_classifier.py
```

This classifier includes:
- Multi-metric uncertainty quantification
- Detailed analysis dashboard
- Better out-of-distribution detection
- Educational content for AIML students

#### Advanced Classifier
```bash
streamlit run advanced_classifier.py
```

#### Original Improved Classifier
```bash
streamlit run app.py
```

## Key Improvements in the Training Approach

### 1. Three-Class Classification
Instead of the original two-class model (cat/dog), the improved model has three classes:
- **Cat**: Images of cats
- **Dog**: Images of dogs
- **Unknown**: Images that are neither cats nor dogs

This explicit "unknown" class allows the model to directly identify out-of-distribution samples.

### 2. Transfer Learning with Fine-tuning
The model uses:
- Pre-trained ResNet50 weights from ImageNet
- Frozen early layers to preserve general features
- Unfrozen last layers for task-specific learning
- Dropout layers for regularization

### 3. Data Augmentation
Training transforms include:
- Random cropping and resizing
- Random horizontal flipping
- Random rotation
- Color jittering
- Normalization

### 4. Improved Loss Function
Cross-entropy loss with:
- Weight decay for regularization
- Learning rate scheduling
- Adam optimizer

## For AIML Students

### Learning Objectives
1. **Uncertainty Quantification**: Understanding how to measure model confidence
2. **Out-of-Distribution Detection**: Techniques for identifying unfamiliar inputs
3. **Transfer Learning**: Leveraging pre-trained models for specific tasks
4. **Data Augmentation**: Improving model generalization through data transformation

### Experiment Ideas
1. Try different architectures (ResNet18, DenseNet, etc.)
2. Experiment with different data augmentation techniques
3. Compare confidence thresholding approaches
4. Implement Monte Carlo Dropout for better uncertainty estimation
5. Try temperature scaling for calibrated probabilities

## Troubleshooting

### Common Issues
1. **Package Installation**: Ensure PyTorch and Torchvision are properly installed
2. **Dataset Structure**: Verify images are in the correct directories
3. **Memory Issues**: Reduce batch size if running out of memory
4. **GPU Usage**: Enable CUDA if you have a compatible GPU

### Environment Verification
Run the test script to verify your environment:
```bash
python test_environment.py
```

## Next Steps

1. **Collect Real Data**: Gather a larger dataset of cat, dog, and unknown images
2. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, etc.
3. **Model Evaluation**: Test the trained model on unseen data
4. **Deployment**: Deploy the enhanced classifier as a web application
5. **Continuous Improvement**: Collect user feedback to further improve the model

## Conclusion

This enhanced classifier addresses the original issue where non-cat/dog images were incorrectly classified as either cats or dogs. By:
1. Including an explicit "unknown" class
2. Implementing multi-metric uncertainty quantification
3. Providing clear feedback for uncertain predictions
4. Creating a more robust training pipeline

The enhanced classifier can now:
- Accurately classify cat and dog images
- Identify when an image doesn't belong to either category
- Provide detailed analysis of prediction confidence
- Offer educational insights into uncertainty quantification