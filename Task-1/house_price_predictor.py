import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class HousePricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        
    def create_sample_dataset(self, n_samples=1000):
        """
        Create a synthetic dataset for house price prediction
        Features: square footage, number of bedrooms, number of bathrooms
        Target: house price
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate features
        square_footage = np.random.normal(2000, 500, n_samples)
        square_footage = np.clip(square_footage, 800, 5000)  # Reasonable range
        
        bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n_samples, p=[0.05, 0.15, 0.35, 0.3, 0.1, 0.05])
        bathrooms = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.25, 0.4, 0.2, 0.05])
        
        # Generate prices based on features with some noise
        # Base price: $100 per sq ft, $10000 per bedroom, $15000 per bathroom
        base_prices = (square_footage * 100 + bedrooms * 10000 + bathrooms * 15000)
        noise = np.random.normal(0, 15000, n_samples)  # Add some randomness
        prices = base_prices + noise
        prices = np.clip(prices, 50000, 1000000)  # Reasonable price range
        
        # Create DataFrame
        data = pd.DataFrame({
            'square_footage': square_footage,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'price': prices
        })
        
        return data
    
    def load_data(self, file_path=None):
        """
        Load data from a CSV file or create sample data
        """
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"File {file_path} not found. Creating sample dataset.")
                self.data = self.create_sample_dataset()
        else:
            print("No data file provided. Creating sample dataset.")
            self.data = self.create_sample_dataset()
            
        return self.data
    
    def explore_data(self):
        """
        Perform basic exploratory data analysis
        """
        if not hasattr(self, 'data'):
            print("No data loaded. Please load data first.")
            return
            
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nData Description:")
        print(self.data.describe())
        print("\nCorrelation Matrix:")
        print(self.data.corr())
        
    def visualize_data(self):
        """
        Create visualizations for the dataset
        """
        if not hasattr(self, 'data'):
            print("No data loaded. Please load data first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price distribution
        axes[0, 0].hist(self.data['price'], bins=30, edgecolor='black')
        axes[0, 0].set_title('Distribution of House Prices')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Square footage vs Price
        axes[0, 1].scatter(self.data['square_footage'], self.data['price'], alpha=0.5)
        axes[0, 1].set_title('Square Footage vs Price')
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Bedrooms vs Price
        axes[1, 0].boxplot([self.data[self.data['bedrooms'] == i]['price'] for i in sorted(self.data['bedrooms'].unique())],
                          labels=sorted(self.data['bedrooms'].unique()))
        axes[1, 0].set_title('Number of Bedrooms vs Price')
        axes[1, 0].set_xlabel('Number of Bedrooms')
        axes[1, 0].set_ylabel('Price ($)')
        
        # Bathrooms vs Price
        axes[1, 1].boxplot([self.data[self.data['bathrooms'] == i]['price'] for i in sorted(self.data['bathrooms'].unique())],
                          labels=sorted(self.data['bathrooms'].unique()))
        axes[1, 1].set_title('Number of Bathrooms vs Price')
        axes[1, 1].set_xlabel('Number of Bathrooms')
        axes[1, 1].set_ylabel('Price ($)')
        
        plt.tight_layout()
        plt.show()
        
    def prepare_features(self):
        """
        Prepare features for training
        """
        if not hasattr(self, 'data'):
            print("No data loaded. Please load data first.")
            return
            
        # Select features and target
        self.X = self.data[['square_footage', 'bedrooms', 'bathrooms']]
        self.y = self.data['price']
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
    def train_model(self):
        """
        Train the linear regression model
        """
        if not hasattr(self, 'X_train'):
            print("Features not prepared. Please prepare features first.")
            return
            
        # Train the model
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print("Model trained successfully!")
        
        # Print model coefficients
        print("\nModel Coefficients:")
        feature_names = ['Square Footage', 'Bedrooms', 'Bathrooms']
        for i, coef in enumerate(self.model.coef_):
            print(f"{feature_names[i]}: ${coef:.2f}")
        print(f"Intercept: ${self.model.intercept_:.2f}")
        
    def evaluate_model(self, show_plots=True):
        """
        Evaluate the trained model
        """
        if not self.is_trained:
            print("Model not trained. Please train the model first.")
            return
            
        # Make predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("\nModel Evaluation:")
        print(f"Training MSE: ${train_mse:.2f}")
        print(f"Testing MSE: ${test_mse:.2f}")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Testing R²: {test_r2:.4f}")
        
        if show_plots:
            try:
                # Ensure we're working with pandas Series for plotting
                y_train_series = pd.Series(self.y_train) if not isinstance(self.y_train, pd.Series) else self.y_train
                y_test_series = pd.Series(self.y_test) if not isinstance(self.y_test, pd.Series) else self.y_test
                
                # Visualize predictions vs actual values
                plt.figure(figsize=(12, 5))
                
                # Training data
                plt.subplot(1, 2, 1)
                plt.scatter(y_train_series, y_train_pred, alpha=0.5)
                min_val = min(float(y_train_series.min()), float(min(y_train_pred)))
                max_val = max(float(y_train_series.max()), float(max(y_train_pred)))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                plt.xlabel('Actual Prices')
                plt.ylabel('Predicted Prices')
                plt.title(f'Training Data (R² = {train_r2:.4f})')
                
                # Testing data
                plt.subplot(1, 2, 2)
                plt.scatter(y_test_series, y_test_pred, alpha=0.5)
                min_val = min(float(y_test_series.min()), float(min(y_test_pred)))
                max_val = max(float(y_test_series.max()), float(max(y_test_pred)))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                plt.xlabel('Actual Prices')
                plt.ylabel('Predicted Prices')
                plt.title(f'Testing Data (R² = {test_r2:.4f})')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Warning: Could not display plots. Error: {e}")
                print("Continuing with the rest of the program...")

    def predict_price(self, square_footage, bedrooms, bathrooms):
        """
        Predict the price of a house given its features
        """
        if not self.is_trained:
            print("Model not trained. Please train the model first.")
            return None
            
        # Create feature array
        features = np.array([[square_footage, bedrooms, bathrooms]])
        
        # Make prediction
        predicted_price = self.model.predict(features)[0]
        
        return predicted_price

def main():
    # Create predictor instance
    predictor = HousePricePredictor()
    
    # Load data (creates sample data since no file is provided)
    data = predictor.load_data()
    
    # Explore data
    predictor.explore_data()
    
    # Prepare features
    predictor.prepare_features()
    
    # Train model
    predictor.train_model()
    
    # Evaluate model (without showing plots to avoid hanging)
    predictor.evaluate_model(show_plots=False)
    
    # Make a sample prediction
    print("\nSample Prediction:")
    sample_price = predictor.predict_price(2000, 3, 2)
    if sample_price:
        print(f"Predicted price for a 2000 sq ft house with 3 bedrooms and 2 bathrooms: ${sample_price:,.2f}")
        
    # Example of how to use with custom data
    print("\nCustom Predictions:")
    print("You can make predictions for any house by calling:")
    print("predictor.predict_price(square_footage, bedrooms, bathrooms)")
    print("Example: predictor.predict_price(1500, 2, 1)")

if __name__ == "__main__":
    main()