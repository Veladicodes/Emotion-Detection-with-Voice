"""
Comprehensive test script for all Emotion Voice API endpoints
Run this to verify all features are working correctly
"""
import requests
import time
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "dev_emotion_voice_api_key_2024"

# Find a test audio file
DATA_DIR = Path(__file__).parent.parent / "Data" / "raw"
TEST_AUDIO = None
for emotion_dir in DATA_DIR.glob("*"):
    if emotion_dir.is_dir():
        audio_files = list(emotion_dir.glob("*.wav"))
        if audio_files:
            TEST_AUDIO = str(audio_files[0])
            break

def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def test_endpoint(name, method, url, **kwargs):
    """Test an endpoint and print results"""
    print(f"\n🧪 Testing: {name}")
    print(f"   {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        
        print(f"   Status: {response.status_code}")
        
        if response.status_code < 400:
            print(f"   ✅ Success")
            data = response.json()
            # Print key info
            if isinstance(data, dict):
                for key, value in list(data.items())[:5]:  # Show first 5 keys
                    if isinstance(value, (str, int, float, bool)):
                        print(f"   • {key}: {value}")
                    elif isinstance(value, dict):
                        print(f"   • {key}: {len(value)} items")
                    elif isinstance(value, list):
                        print(f"   • {key}: {len(value)} items")
        else:
            print(f"   ❌ Failed: {response.text}")
        
        return response.status_code < 400
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print_section("Emotion Voice API - Comprehensive Test Suite")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Server is not running!")
            print("   Start server with: uvicorn main:app --reload")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Start server with: uvicorn main:app --reload")
        sys.exit(1)
    
    print("✅ Server is running!")
    
    results = {}
    
    # Test Public Endpoints
    print_section("PUBLIC ENDPOINTS")
    
    results["root"] = test_endpoint(
        "Root Endpoint",
        "GET",
        f"{BASE_URL}/"
    )
    
    results["health"] = test_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/health"
    )
    
    results["status"] = test_endpoint(
        "System Status",
        "GET",
        f"{BASE_URL}/api/status"
    )
    
    results["gpu"] = test_endpoint(
        "GPU Information",
        "GET",
        f"{BASE_URL}/api/gpu"
    )
    
    # Test Inference Endpoint
    print_section("INFERENCE ENDPOINT")
    
    if TEST_AUDIO:
        print(f"   Using test file: {TEST_AUDIO}")
        results["predict"] = test_endpoint(
            "Predict Emotion",
            "POST",
            f"{BASE_URL}/api/predict",
            files={"file": open(TEST_AUDIO, "rb")}
        )
    else:
        print("   ⚠️  No test audio file found. Skipping prediction test.")
        results["predict"] = False
    
    # Test Admin Endpoints
    print_section("ADMIN ENDPOINTS (Protected)")
    
    headers = {"x-api-key": API_KEY}
    
    results["models"] = test_endpoint(
        "List Models",
        "GET",
        f"{BASE_URL}/api/models",
        headers=headers
    )
    
    results["metrics"] = test_endpoint(
        "Get Metrics",
        "GET",
        f"{BASE_URL}/api/metrics",
        headers=headers
    )
    
    # Test API Key Protection
    print_section("SECURITY TESTS")
    
    print("\n🧪 Testing: API Key Protection (should fail)")
    response = requests.get(f"{BASE_URL}/api/models")
    if response.status_code == 401:
        print("   ✅ Correctly rejected request without API key")
        results["security"] = True
    else:
        print(f"   ❌ Expected 401, got {response.status_code}")
        results["security"] = False
    
    print("\n🧪 Testing: Invalid API Key (should fail)")
    response = requests.get(f"{BASE_URL}/api/models", headers={"x-api-key": "invalid_key"})
    if response.status_code == 403:
        print("   ✅ Correctly rejected request with invalid API key")
    else:
        print(f"   ❌ Expected 403, got {response.status_code}")
        results["security"] = False
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n📊 Results: {passed}/{total} tests passed\n")
    
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("🎉 All tests passed! API is fully functional.")
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check output above.")
    
    print("=" * 70)
    
    # Next steps
    print("\n📚 Next Steps:")
    print("   • View interactive docs: http://127.0.0.1:8000/docs")
    print("   • Read API reference: backend/API_REFERENCE.md")
    print("   • Check logs: backend/logs/requests.csv")
    print("   • Monitor GPU: curl http://127.0.0.1:8000/api/gpu")
    print()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

