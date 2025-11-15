import requests
import json

def test_index():
    """Test the index route"""
    try:
        response = requests.get('http://127.0.0.1:5000/')
        print(f"Index route status code: {response.status_code}")
        print(f"Index route response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error testing index route: {e}")

def test_predict():
    """Test the predict route with demo data"""
    try:
        # Since we're in demo mode, we can send a simple request
        response = requests.post('http://127.0.0.1:5000/predict', 
                               files={'image': ('test.jpg', b'test image data', 'image/jpeg')})
        print(f"Predict route status code: {response.status_code}")
        print(f"Predict route response: {response.json()}")
    except Exception as e:
        print(f"Error testing predict route: {e}")

def test_predict_frame():
    """Test the predict_frame route with demo data"""
    try:
        # Send a simple JSON request
        response = requests.post('http://127.0.0.1:5000/predict_frame', 
                               json={'image': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBAQE...'})
        print(f"Predict frame route status code: {response.status_code}")
        print(f"Predict frame route response: {response.json()}")
    except Exception as e:
        print(f"Error testing predict_frame route: {e}")

if __name__ == "__main__":
    print("Testing Flask application endpoints...")
    test_index()
    test_predict()
    test_predict_frame()
    print("Testing complete!")