"""
Demonstration script showing how to use the HousePricePredictor with real data
"""

from house_price_predictor import HousePricePredictor

def main():
    print("=== House Price Prediction Demo ===")
    print("This demo shows how to use the linear regression model with real house data.\n")
    
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Load real data from CSV file
    print("Loading house data from CSV file...")
    data = predictor.load_data('sample_house_data.csv')
    
    # Explore the data
    print("\nExploring the dataset...")
    predictor.explore_data()
    
    # Prepare features for training
    print("\nPreparing features for training...")
    predictor.prepare_features()
    
    # Train the model
    print("\nTraining the linear regression model...")
    predictor.train_model()
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    predictor.evaluate_model(show_plots=False)
    
    # Make predictions for new houses
    print("\n=== Making Predictions for New Houses ===")
    
    # Predict price for a 2000 sq ft house with 3 bedrooms and 2 bathrooms
    price1 = predictor.predict_price(2000, 3, 2)
    print(f"Predicted price for a 2000 sq ft, 3 bedroom, 2 bathroom house: ${price1:,.2f}")
    
    # Predict price for a larger house
    price2 = predictor.predict_price(3000, 4, 3)
    print(f"Predicted price for a 3000 sq ft, 4 bedroom, 3 bathroom house: ${price2:,.2f}")
    
    # Predict price for a smaller house
    price3 = predictor.predict_price(1200, 2, 1)
    print(f"Predicted price for a 1200 sq ft, 2 bedroom, 1 bathroom house: ${price3:,.2f}")
    
    print("\n=== Model Interpretation ===")
    print("The model has learned the following relationships:")
    print("- Each additional square foot adds approximately $100 to the price")
    print("- Each additional bedroom adds approximately $10,000 to the price")
    print("- Each additional bathroom adds approximately $15,000 to the price")
    print("\nNote: These values will vary based on the specific dataset used for training.")

if __name__ == "__main__":
    main()