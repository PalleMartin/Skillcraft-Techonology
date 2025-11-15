"""
Example script demonstrating how to use the HousePricePredictor class
with custom data or a CSV file.
"""

from house_price_predictor import HousePricePredictor
import pandas as pd

def example_with_custom_data():
    """Example of using the model with custom data"""
    print("=== Example: Using the model with custom data ===")
    
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Load sample data (in practice, you would load your own data)
    data = predictor.load_data()
    
    # Prepare features and train model
    predictor.prepare_features()
    predictor.train_model()
    
    # Make predictions for different houses
    print("\nPredictions for sample houses:")
    
    # Example 1: Small house
    price1 = predictor.predict_price(1200, 2, 1)
    print(f"1200 sq ft, 2 bedrooms, 1 bathroom: ${price1:,.2f}")
    
    # Example 2: Medium house
    price2 = predictor.predict_price(2000, 3, 2)
    print(f"2000 sq ft, 3 bedrooms, 2 bathrooms: ${price2:,.2f}")
    
    # Example 3: Large house
    price3 = predictor.predict_price(3500, 5, 4)
    print(f"3500 sq ft, 5 bedrooms, 4 bathrooms: ${price3:,.2f}")

def example_with_csv_file():
    """Example of using the model with a CSV file"""
    print("\n\n=== Example: Using the model with a CSV file ===")
    
    # Create a sample CSV file for demonstration
    sample_data = pd.DataFrame({
        'square_footage': [1500, 2200, 2800, 1800, 3200],
        'bedrooms': [2, 3, 4, 3, 5],
        'bathrooms': [1, 2, 3, 2, 4],
        'price': [200000, 320000, 450000, 280000, 550000]  # This would be unknown in real scenario
    })
    
    sample_data.to_csv('sample_houses.csv', index=False)
    print("Created sample CSV file: sample_houses.csv")
    
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Load data from CSV file
    data = predictor.load_data('sample_houses.csv')
    print(f"Loaded {len(data)} houses from CSV file")
    
    # Prepare features and train model
    predictor.prepare_features()
    predictor.train_model()
    
    # Make a prediction for a new house
    new_price = predictor.predict_price(2500, 4, 3)
    print(f"\nPredicted price for a 2500 sq ft, 4 bedroom, 3 bathroom house: ${new_price:,.2f}")

if __name__ == "__main__":
    # Run examples
    example_with_custom_data()
    example_with_csv_file()
    
    print("\n=== Usage Summary ===")
    print("1. Import HousePricePredictor from house_price_predictor")
    print("2. Create an instance: predictor = HousePricePredictor()")
    print("3. Load your data: predictor.load_data('your_file.csv') or predictor.load_data() for sample data")
    print("4. Prepare features: predictor.prepare_features()")
    print("5. Train the model: predictor.train_model()")
    print("6. Make predictions: predictor.predict_price(square_footage, bedrooms, bathrooms)")