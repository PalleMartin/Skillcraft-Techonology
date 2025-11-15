# Task 1: Full-Stack House Price Prediction using Linear Regression

This task implements a full-stack linear regression model with Flask backend and HTML/CSS/JavaScript frontend to predict house prices based on square footage, number of bedrooms, and number of bathrooms. It is part of the SkillCraft-ML project.

## Features

- Creates a synthetic dataset for house price prediction
- Implements linear regression using scikit-learn
- Provides data exploration and visualization
- Evaluates model performance with MSE and R² metrics
- Allows prediction of house prices for new data

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Navigate to the Task-1 directory:
   ```
   cd Task-1
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Flask web application:
```
python app.py
```

This will start a web server where you can:
1. View the model information
2. Predict house prices based on input features
3. See model coefficients and performance metrics

## Model Details

The linear regression model takes the following features as input:
- Square footage
- Number of bedrooms
- Number of bathrooms

The model outputs the predicted price of the house.

## Custom Dataset

To use your own dataset, create a CSV file with the following columns:
- `square_footage`: Numeric
- `bedrooms`: Integer
- `bathrooms`: Integer
- `price`: Numeric (target variable)

Then modify the `load_data()` method to load your CSV file.

## License

This project is open source and available under the MIT License.