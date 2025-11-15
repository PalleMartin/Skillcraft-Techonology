# Enhanced Cat vs Dog Classifier

This project contains an enhanced version of the cat vs dog image classifier that better handles out-of-distribution samples.

## Files in this Project

### Main Application Files

1. **[app.py](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/app.py)** - Original application with basic confidence thresholding
2. **[advanced_classifier.py](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/advanced_classifier.py)** - Advanced classifier with multi-metric uncertainty quantification
3. **[final_enhanced_classifier.py](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/final_enhanced_classifier.py)** - Final enhanced classifier with detailed analysis dashboard
4. **[train_improved_model.py](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/train_improved_model.py)** - Training script demonstrating how to create a more robust model

### Documentation Files

1. **[IMPROVEMENTS_SUMMARY.md](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/IMPROVEMENTS_SUMMARY.md)** - Detailed summary of all improvements made
2. **[README.md](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/README.md)** - This file

### Model Files

1. **[model_trained.pth](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/model_trained.pth)** - Pre-trained model weights
2. **[requirements.txt](file:///Users/gayatri/Documents/Martin%20Palle%20/INTERNSHIPS/SkillCraft-ML/Task-3/cats_vs_dogs_prediction_app-main/requirements.txt)** - Python dependencies

## How to Run the Applications

### Prerequisites
Make sure you have installed all required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Final Enhanced Classifier (Recommended)
```bash
streamlit run final_enhanced_classifier.py
```

### Running Other Versions
```bash
# Original improved version
streamlit run app.py

# Advanced classifier
streamlit run advanced_classifier.py
```

## Key Improvements

### Better Out-of-Distribution Detection
The enhanced classifiers can now detect when an image doesn't belong to either the "cat" or "dog" category and will flag it as uncertain rather than forcing a classification.

### Multi-Metric Reliability Assessment
Instead of relying on a single confidence metric, the enhanced classifier uses three criteria:
1. **High Confidence**: Maximum probability > 70%
2. **Low Entropy**: Normalized entropy < 70% of maximum
3. **Clear Margin**: Difference between top predictions > 0.2

### Detailed Analysis Dashboard
The final enhanced classifier provides:
- Visual probability charts
- Multiple uncertainty metrics
- Reliability assessment progress bar
- Detailed criteria analysis

## Testing the Improvements

When you upload images that are not cats or dogs (such as human images or leaf images), you should see:

1. **Lower confidence scores**
2. **Higher entropy values**
3. **Smaller margins between predictions**
4. **"Likely Not a Cat or Dog" warning**
5. **Detailed explanations about why the prediction is unreliable**

## For AIML Students

The project demonstrates several important machine learning concepts:
1. **Uncertainty Quantification**: How to measure and interpret model uncertainty
2. **Out-of-Distribution Detection**: Techniques for identifying when inputs don't match training data
3. **Multi-Criteria Decision Making**: Using multiple metrics for more robust predictions
4. **Transfer Learning**: Using pre-trained models for specific tasks

## Future Improvements

To further enhance the classifier, consider:
1. Training with an explicit "unknown" class
2. Implementing Monte Carlo Dropout for better uncertainty estimation
3. Using temperature scaling for better calibrated probabilities
4. Adding more diverse out-of-distribution samples during training

## Troubleshooting

If you encounter issues:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that the model file exists: `model_trained.pth`
3. Ensure you have internet connection for initial model download
4. If you see SSL certificate errors, try: `pip install --upgrade certifi`

## Author
Martin Palle - AIML Student