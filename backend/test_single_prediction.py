"""
Quick single prediction test
Tests one audio file and displays results
"""

import requests
import sys
import os

# Configuration
API_URL = "http://127.0.0.1:8000/api/predict"
DEFAULT_AUDIO = r"D:\Emotion Voice\Data\raw\happy\sample.wav"

def test_prediction(audio_path):
    """Test single prediction"""
    
    if not os.path.exists(audio_path):
        print(f"❌ File not found: {audio_path}")
        return
    
    print("=" * 70)
    print("  EMOTION PREDICTION TEST")
    print("=" * 70)
    print(f"Audio: {audio_path}")
    print(f"API: {API_URL}")
    print("\nUploading...")
    
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': (os.path.basename(audio_path), f, 'audio/wav')}
            response = requests.post(API_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n" + "=" * 70)
            print("  PREDICTION RESULTS")
            print("=" * 70)
            print(f"\n🎭 Emotion:     {data['emotion'].upper()}")
            print(f"📊 Confidence:  {data['confidence']:.2%}")
            print(f"⚡ Latency:     {data['latency_ms']} ms")
            print(f"💻 Device:      {data['device']}")
            print(f"🤖 Model:       {data['model']}")
            print(f"⏱️  Duration:    {data['audio_duration_seconds']} seconds")
            
            if 'warning' in data:
                print(f"\n⚠️  {data['warning']}")
            
            print(f"\n" + "-" * 70)
            print("  EMOTION PROBABILITIES")
            print("-" * 70)
            
            classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            scores = data['raw_scores']
            
            # Sort by score
            emotion_scores = list(zip(classes, scores))
            emotion_scores.sort(key=lambda x: x[1], reverse=True)
            
            for emotion, score in emotion_scores:
                bar_length = int(score * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                print(f"{emotion:10s} {score:.4f} [{bar}] {score*100:.2f}%")
            
            print("\n✅ Test completed successfully!")
            
        else:
            print(f"\n❌ Error {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("\nMake sure the server is running:")
        print("  cd backend")
        print("  uvicorn main:app --reload")
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    test_prediction(audio)

