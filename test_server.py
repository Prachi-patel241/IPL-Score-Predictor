"""
Quick test script to verify the server is working
"""
import requests
import json

# Test the server
try:
    # Test if server is running
    response = requests.get('http://localhost:5000', timeout=5)
    print(f"✓ Server is running (Status: {response.status_code})")
except requests.exceptions.ConnectionError:
    print("✗ Server is not running. Please start it with: python app.py")
except Exception as e:
    print(f"✗ Error: {e}")

# Test the predict endpoint with sample data
try:
    test_data = {
        "batting_team": "MI",
        "bowling_team": "CSK",
        "batsman": "Virat Kohli",
        "bowler": "Jasprit Bumrah",
        "stadium": "Wankhede Stadium",
        "current_over": 10.0,
        "current_score": 100,
        "current_wickets": 2,
        "pitch_moisture": 50,
        "ball_type": "regular"
    }
    
    response = requests.post('http://localhost:5000/api/predict', 
                           json=test_data, 
                           timeout=5)
    result = response.json()
    if result.get('success'):
        print(f"✓ Predict endpoint working (Predicted: {result.get('predicted_runs')} runs)")
    else:
        print(f"✗ Predict endpoint error: {result.get('error')}")
except Exception as e:
    print(f"✗ Predict endpoint test failed: {e}")

# Test the final score endpoint
try:
    test_data = {
        "batting_team": "MI",
        "bowling_team": "CSK",
        "stadium": "Wankhede Stadium",
        "current_over": 10.0,
        "current_score": 100,
        "current_wickets": 2,
        "pitch_moisture": 50
    }
    
    response = requests.post('http://localhost:5000/api/predict-final', 
                           json=test_data, 
                           timeout=5)
    result = response.json()
    if result.get('success'):
        print(f"✓ Final score endpoint working (Predicted: {result.get('final_score')})")
    else:
        print(f"✗ Final score endpoint error: {result.get('error')}")
except Exception as e:
    print(f"✗ Final score endpoint test failed: {e}")

