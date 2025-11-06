"""
Simple script to start the server and show any errors
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    print("=" * 70)
    print("Starting Emotion Voice API Server...")
    print("=" * 70)
    
    # Import main app
    from main import app
    print("✅ Main app imported successfully")
    
    # Start uvicorn
    import uvicorn
    print("✅ Starting server on http://0.0.0.0:8000")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    
except Exception as e:
    print("=" * 70)
    print(f"❌ ERROR: {e}")
    print("=" * 70)
    import traceback
    traceback.print_exc()
    sys.exit(1)

