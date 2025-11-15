#!/bin/bash
# Script to run the final enhanced cat vs dog classifier

echo "Starting Enhanced Cat vs Dog Classifier..."
echo "========================================"

# Check if required files exist
if [ ! -f "model_trained.pth" ]; then
    echo "Warning: Model file not found. It will be downloaded automatically when needed."
fi

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found!"
    exit 1
fi

# Install/update dependencies
echo "Checking dependencies..."
pip install -r requirements.txt

# Run the final enhanced classifier
echo "Launching the Enhanced Cat vs Dog Classifier..."
echo "Access the application in your browser at: http://localhost:8501"
streamlit run final_enhanced_classifier.py