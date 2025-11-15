import os
from flask import Flask, render_template, request, jsonify
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from house_price_predictor import HousePricePredictor

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')

# Initialize Flask app with explicit template directory
app = Flask(__name__, template_folder=template_dir)

# Initialize the predictor
predictor = HousePricePredictor()

# Load and train the model
print("Loading and training the model...")
try:
    # Try to load the sample data file
    sample_data_path = os.path.join(current_dir, 'sample_house_data.csv')
    if os.path.exists(sample_data_path):
        data = predictor.load_data(sample_data_path)
    else:
        # If no data file exists, create sample data
        data = predictor.load_data()
    
    predictor.prepare_features()
    predictor.train_model()
    print("Model loaded and trained successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a simple model with default data if there's an error
    data = predictor.create_sample_dataset(100)
    predictor.prepare_features()
    predictor.train_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        square_footage = float(request.form['square_footage'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        
        # Make prediction
        predicted_price = predictor.predict_price(square_footage, bedrooms, bathrooms)
        
        if predicted_price is not None:
            return jsonify({
                'success': True,
                'price': f"${predicted_price:,.2f}"
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            })
    except Exception as e:
        # Ensure we always return a JSON response
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/model_info')
def model_info():
    # Get model coefficients if available
    try:
        if hasattr(predictor, 'model') and predictor.is_trained:
            coefficients = {
                'square_footage': f"${predictor.model.coef_[0]:.2f}",
                'bedrooms': f"${predictor.model.coef_[1]:.2f}",
                'bathrooms': f"${predictor.model.coef_[2]:.2f}",
                'intercept': f"${predictor.model.intercept_:.2f}"
            }
            return jsonify({
                'success': True,
                'coefficients': coefficients
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Model not trained'
            })
    except Exception as e:
        # Ensure we always return a JSON response
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add error handlers to ensure JSON responses
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
