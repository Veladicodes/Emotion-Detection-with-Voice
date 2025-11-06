"""
BACKEND ALIGNMENT TEST
Verify that backend preprocessing and inference match the training pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.audio_preprocess import verify_backend_alignment
from pathlib import Path
import numpy as np

def test_alignment():
    """Test alignment between training and inference"""
    
    print("\n" + "="*70)
    print("TESTING BACKEND ALIGNMENT WITH TRAINING PIPELINE")
    print("="*70)
    
    # Find a sample file
    base_dir = Path(r"D:\Emotion Voice")
    
    # Test with happy emotion
    npy_files = list((base_dir / "Data" / "processed" / "mel_npy" / "happy").glob("*.npy"))
    
    if not npy_files:
        print("❌ No preprocessed .npy files found!")
        print("   Run the training notebook first to generate mel spectrograms.")
        return False
    
    # Use first file
    npy_path = npy_files[0]
    audio_name = npy_path.stem + ".wav"
    audio_path = base_dir / "Data" / "raw" / "happy" / audio_name
    
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        return False
    
    print(f"\nTesting with files:")
    print(f"  NPY (training): {npy_path.name}")
    print(f"  WAV (inference): {audio_path.name}")
    
    # Run verification
    diff = verify_backend_alignment(str(npy_path), str(audio_path))
    
    # Test with a few more samples
    print("\n" + "="*70)
    print("TESTING MULTIPLE SAMPLES")
    print("="*70)
    
    differences = []
    emotions_tested = []
    
    for emotion in ['happy', 'sad', 'angry']:
        npy_dir = base_dir / "Data" / "processed" / "mel_npy" / emotion
        audio_dir = base_dir / "Data" / "raw" / emotion
        
        if not npy_dir.exists():
            continue
        
        npy_files = list(npy_dir.glob("*.npy"))[:3]  # Test 3 samples per emotion
        
        for npy_file in npy_files:
            audio_file = audio_dir / (npy_file.stem + ".wav")
            if audio_file.exists():
                print(f"\nTesting {emotion}/{npy_file.name}...")
                diff = verify_backend_alignment(str(npy_file), str(audio_file))
                differences.append(diff)
                emotions_tested.append(emotion)
    
    # Summary
    if differences:
        avg_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        print("\n" + "="*70)
        print("ALIGNMENT SUMMARY")
        print("="*70)
        print(f"  Samples tested: {len(differences)}")
        print(f"  Emotions tested: {set(emotions_tested)}")
        print(f"  Average difference: {avg_diff:.6f}")
        print(f"  Maximum difference: {max_diff:.6f}")
        
        if avg_diff < 0.01:
            print(f"\n  SUCCESS: Backend is perfectly aligned with training!")
            print(f"     Predictions should match validation accuracy (~74%)")
            return True
        elif avg_diff < 0.1:
            print(f"\n  GOOD: Alignment is acceptable")
            return True
        else:
            print(f"\n  WARNING: Significant differences detected")
            print(f"     Review preprocessing parameters")
            return False
    
    return False


def verify_config():
    """Verify configuration matches training"""
    import json
    
    print("\n" + "="*70)
    print("VERIFYING CONFIGURATION")
    print("="*70)
    
    config_path = Path(__file__).parent / "config" / "training_config.json"
    
    if not config_path.exists():
        print("❌ Training config not found!")
        return False
    
    with open(config_path) as f:
        config = json.load(f)
    
    from utils.audio_preprocess import get_preprocessing_params
    backend_params = get_preprocessing_params()
    training_params = config["preprocessing"]
    
    print("\nComparing parameters:")
    mismatches = []
    
    for key in ["SAMPLE_RATE", "N_MELS", "N_FFT", "HOP_LENGTH", "TARGET_SIZE"]:
        training_val = training_params[key]
        backend_val = backend_params[key]
        
        match = training_val == backend_val
        symbol = "[OK]" if match else "[FAIL]"
        
        print(f"  {symbol} {key}: training={training_val}, backend={backend_val}")
        
        if not match:
            mismatches.append(key)
    
    if mismatches:
        print(f"\nERROR: {len(mismatches)} parameter mismatches found!")
        return False
    else:
        print(f"\nSUCCESS: All parameters match!")
        return True


if __name__ == "__main__":
    print("\nBACKEND ALIGNMENT TEST SUITE\n")
    
    # Test 1: Config verification
    config_ok = verify_config()
    
    # Test 2: Alignment verification
    alignment_ok = test_alignment()
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    if config_ok and alignment_ok:
        print("SUCCESS: Backend is correctly aligned with training pipeline!")
        print("   Predictions should match validation accuracy (~74%)")
        print("\nYou can now confidently use the system for inference.")
    else:
        print("WARNING: Some alignment issues detected.")
        print("   Review the output above and fix discrepancies.")
    
    print("="*70 + "\n")

