@echo off
REM Script to run the final enhanced cat vs dog classifier on Windows

echo Starting Enhanced Cat vs Dog Classifier...
echo ========================================

REM Check if required files exist
if not exist "model_trained.pth" (
    echo Warning: Model file not found. It will be downloaded automatically when needed.
)

if not exist "requirements.txt" (
    echo Error: requirements.txt not found!
    exit /b 1
)

REM Install/update dependencies
echo Checking dependencies...
pip install -r requirements.txt

REM Run the final enhanced classifier
echo Launching the Enhanced Cat vs Dog Classifier...
echo Access the application in your browser at: http://localhost:8501
streamlit run final_enhanced_classifier.py

pause