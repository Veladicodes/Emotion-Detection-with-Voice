"""
Phase 4: API Testing Script
Tests all endpoints of the Emotion Voice API
"""

import requests
import os
from pathlib import Path

# API Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = "your-secret-api-key-here"  # Update with your actual API key

# Test audio file
TEST_AUDIO = r"D:\Emotion Voice\Data\raw\happy\sample.wav"


def print_section(title):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_root():
    """Test root endpoint"""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_status():
    """Test status endpoint"""
    print_section("Testing Status Endpoint")
    response = requests.get(f"{BASE_URL}/api/status")
    print(f"Status: {response.status_code}")
    print(f"Response:")
    data = response.json()
    for key, value in data.items():
        print(f"  {key}: {value}")
    return response.status_code == 200


def test_gpu():
    """Test GPU status endpoint"""
    print_section("Testing GPU Status Endpoint")
    response = requests.get(f"{BASE_URL}/api/gpu")
    print(f"Status: {response.status_code}")
    print(f"Response:")
    data = response.json()
    for key, value in data.items():
        print(f"  {key}: {value}")
    return response.status_code == 200


def test_predict():
    """Test prediction endpoint with audio file"""
    print_section("Testing Prediction Endpoint")
    
    if not os.path.exists(TEST_AUDIO):
        print(f"⚠️  Test audio file not found: {TEST_AUDIO}")
        print("Please update TEST_AUDIO path in test_api.py")
        return False
    
    print(f"Uploading: {TEST_AUDIO}")
    
    with open(TEST_AUDIO, 'rb') as f:
        files = {'file': (os.path.basename(TEST_AUDIO), f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/api/predict", files=files)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nPrediction Results:")
        print(f"  Emotion: {data.get('emotion')}")
        print(f"  Confidence: {data.get('confidence'):.2%}")
        print(f"  Latency: {data.get('latency_ms')} ms")
        print(f"  Device: {data.get('device')}")
        print(f"  Model: {data.get('model')}")
        print(f"  Audio Duration: {data.get('audio_duration_seconds')} seconds")
        
        if 'warning' in data:
            print(f"\n⚠️  Warning: {data['warning']}")
        
        print(f"\nRaw Scores:")
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for cls, score in zip(classes, data.get('raw_scores', [])):
            print(f"  {cls:10s}: {score:.4f} ({score*100:.2f}%)")
        
        return True
    else:
        print(f"Error: {response.text}")
        return False


def test_metrics():
    """Test metrics endpoint"""
    print_section("Testing Metrics Endpoint")
    response = requests.get(f"{BASE_URL}/api/metrics")
    print(f"Status: {response.status_code}")
    print(f"Response:")
    data = response.json()
    
    if 'error' in data:
        print(f"  {data['error']}")
    else:
        print(f"  Total Inferences: {data.get('total_inferences')}")
        print(f"  Most Frequent Emotion: {data.get('most_frequent_emotion')}")
        print(f"  Latency Stats: {data.get('latency_stats')}")
        print(f"  Model Info: {data.get('model_info')}")
    
    return response.status_code == 200


def test_list_models():
    """Test list models endpoint (protected)"""
    print_section("Testing List Models Endpoint (Protected)")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(f"{BASE_URL}/api/models", headers=headers)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"Available Models: {len(models)}")
        for model in models:
            print(f"\n  Name: {model['name']}")
            print(f"  Location: {model['location']}")
            print(f"  Size: {model['size_mb']} MB")
            print(f"  Current: {model['is_current']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code in [200, 403]  # 403 if API key wrong


def run_all_tests():
    """Run all API tests"""
    print("\n" + "=" * 70)
    print("  EMOTION VOICE API - PHASE 4 TESTING")
    print("=" * 70)
    print(f"\nBase URL: {BASE_URL}")
    print(f"Test Audio: {TEST_AUDIO}")
    
    results = {
        "Root": test_root(),
        "Health": test_health(),
        "Status": test_status(),
        "GPU": test_gpu(),
        "Predict": test_predict(),
        "Metrics": test_metrics(),
        "List Models": test_list_models()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the server is running:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
