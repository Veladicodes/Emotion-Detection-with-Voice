"""
Phase 4: TorchScript Model Loader
Loads the exported ensemble model from Phase 3 using torch.jit.load
"""

import torch
import os
import warnings
from pathlib import Path
import time

# Class labels used in training (keep exactly the same order as training)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === PHASE 3 TORCHSCRIPT MODEL PATH ===
MODEL_PATH = r"D:\Emotion Voice\phase3_outputs\final_export\best_efficientnet_v2_m+vgg16+densenet121.pt"

# Global state for model management
_MODEL_LOADED = False
_CURRENT_MODEL = None
_CURRENT_DEVICE = None
_CURRENT_MODEL_PATH = MODEL_PATH
_CURRENT_MODEL_NAME = "best_efficientnet_v2_m+vgg16+densenet121.pt"
_MODEL_LOAD_TIME = None
_SERVER_START_TIME = time.time()

def load_model():
    """
    Load TorchScript model exported from Phase 3
    Returns: (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(_CURRENT_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {_CURRENT_MODEL_PATH}")
    
    print(f"Loading TorchScript model from: {_CURRENT_MODEL_PATH}")
    model = torch.jit.load(_CURRENT_MODEL_PATH, map_location=device)
    model.eval()
    model.to(device)
    
    print(f"SUCCESS: Model loaded successfully on {device}")
    return model, device


def get_model():
    """Load model or return None if not available (dummy mode)"""
    global _MODEL_LOADED, _CURRENT_MODEL, _CURRENT_DEVICE, _CURRENT_MODEL_PATH, _CURRENT_MODEL_NAME, _MODEL_LOAD_TIME
    
    # Return cached model if already loaded
    if _MODEL_LOADED and _CURRENT_MODEL is not None:
        return _CURRENT_MODEL, _CURRENT_DEVICE, CLASSES
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _CURRENT_DEVICE = device
    
    # Check if model path exists
    if not os.path.exists(_CURRENT_MODEL_PATH):
        print("=" * 70)
        print("WARNING [model_loader] MODEL NOT FOUND - Running in DUMMY MODE")
        print(f"    Expected path: {_CURRENT_MODEL_PATH}")
        print("    The server will start but return random predictions.")
        print("    Export your Phase 3 model first, then restart.")
        print("=" * 70)
        _MODEL_LOADED = False
        _CURRENT_MODEL = None
        return None, device, CLASSES
    
    # Try to load TorchScript model
    try:
        start_time = time.time()
        model = torch.jit.load(_CURRENT_MODEL_PATH, map_location=device)
        model.eval()
        model.to(device)
        load_time = time.time() - start_time
        _MODEL_LOAD_TIME = load_time
        
        print("=" * 70)
        print(f"SUCCESS [model_loader] TorchScript Model loaded successfully")
        print(f"    Path: {_CURRENT_MODEL_PATH}")
        print(f"    Name: {_CURRENT_MODEL_NAME}")
        print(f"    Device: {device}")
        print(f"    Load time: {load_time:.2f}s")
        print("=" * 70)
        _MODEL_LOADED = True
        _CURRENT_MODEL = model
        return model, device, CLASSES
    
    except Exception as e:
        print("=" * 70)
        print(f"WARNING [model_loader] Failed to load TorchScript model: {e}")
        print("    Running in DUMMY MODE with random predictions.")
        print("=" * 70)
        _MODEL_LOADED = False
        _CURRENT_MODEL = None
        return None, device, CLASSES


def predict_emotion(model, mel_tensor, device, classes, tracer=None):
    """
    PHASE 5 TIER-0: Predict emotion with comprehensive tracing and validation
    
    Args:
        model: TorchScript model
        mel_tensor: preprocessed mel spectrogram (1, 128, 128) or (B, 1, 128, 128)
        device: torch device
        classes: list of emotion labels
        tracer: InferenceTracer instance for logging
    Returns:
        (emotion_label, confidence, raw_probabilities)
    """
    global _MODEL_LOADED
    import time
    import numpy as np
    
    # DUMMY MODE: return random prediction when no model loaded
    if model is None or not _MODEL_LOADED:
        import random
        # Return random prediction with warning
        idx = random.randint(0, len(classes) - 1)
        probs = np.random.dirichlet(np.ones(len(classes)))
        
        if tracer:
            tracer.log_error("MODEL_INFERENCE", "Model not loaded - returning random predictions")
        
        return classes[idx], float(probs[idx]), probs.tolist()
    
    # REAL MODEL: perform actual inference
    inference_start = time.perf_counter()
    model.eval()
    
    with torch.no_grad():
        x = mel_tensor.to(device)
        
        # Ensure correct dimensions: (B, C, H, W) = (1, 1, 128, 128)
        # Input might be (1, 128, 128) or (B, 1, 128, 128)
        if x.dim() == 3:
            # (1, 128, 128) -> (1, 1, 128, 128)
            x = x.unsqueeze(0)
        elif x.dim() == 5:
            # (1, 1, 1, 128, 128) -> (1, 1, 128, 128)
            x = x.squeeze(0)
        
        # PHASE 5 TIER-0: Input validation and tracing
        has_nan = torch.isnan(x).any().item()
        has_inf = torch.isinf(x).any().item()
        
        if tracer:
            tracer.log_model_input(
                input_shape=tuple(x.shape),
                input_dtype=str(x.dtype),
                device=str(device),
                has_nan=has_nan,
                has_inf=has_inf,
                min_val=float(x.min()),
                max_val=float(x.max())
            )
        
        # Check for NaNs or Infs
        if has_nan:
            error_msg = "NaNs detected in model input tensor - preprocessing error"
            print(f"[TRACE] ❌ {error_msg}")
            if tracer:
                tracer.log_error("MODEL_INPUT_VALIDATION", error_msg)
            raise ValueError(error_msg)
        
        if has_inf:
            error_msg = "Infs detected in model input tensor - preprocessing error"
            print(f"[TRACE] ❌ {error_msg}")
            if tracer:
                tracer.log_error("MODEL_INPUT_VALIDATION", error_msg)
            raise ValueError(error_msg)
        
        # Verify shape
        expected_shape = (1, 1, 128, 128)
        if tuple(x.shape) != expected_shape:
            error_msg = f"Input shape {x.shape} does not match expected {expected_shape}"
            print(f"[TRACE] ⚠️  {error_msg}")
            if tracer:
                tracer.log_error("MODEL_INPUT_VALIDATION", error_msg)
        
        print(f"[TRACE] Model input shape: {x.shape} | Device: {x.device} | dtype: {x.dtype}")
        print(f"[TRACE] Input range: [{x.min():.6f}, {x.max():.6f}]")
        
        # TorchScript inference
        output = model(x)
        
        # BACKEND ALIGNMENT FIX: Ensemble already applies softmax internally
        # The TorchScript model outputs averaged probabilities, not raw logits
        # Applying softmax again causes double-softmax distortion
        probs = output[0].cpu()
        probs_np = probs.numpy()
        
        # Normalize to ensure sum = 1.0 (safe normalization)
        probs = probs / probs.sum()
        probs_np = probs.numpy()
        
        print(f"[TRACE] Model output (probabilities): {probs_np}")
        print(f"[TRACE] Sum check: {probs_np.sum():.6f} (should be 1.0)")
        
        # Get prediction
        conf, idx = torch.max(probs, 0)
        predicted_emotion = classes[int(idx)]
        
        inference_time_ms = (time.perf_counter() - inference_start) * 1000
        
        print(f"[TRACE] Predicted: {predicted_emotion} ({float(conf):.4f}) in {inference_time_ms:.2f}ms")
        
        # PHASE 5 TIER-0: Log complete inference results
        if tracer:
            tracer.log_model_inference(
                model_name=_CURRENT_MODEL_NAME,
                device=str(device),
                logits=probs_np,  # These are probabilities, not logits (model applies softmax)
                probabilities=probs_np,
                predicted_class=int(idx),
                predicted_emotion=predicted_emotion,
                confidence=float(conf),
                inference_time_ms=inference_time_ms
            )
        
        return predicted_emotion, float(conf), probs_np.tolist()


# ============================================================================
# Model Management Functions
# ============================================================================

def get_available_models():
    """
    List all available model checkpoints
    Returns: list of dict with model info
    """
    models = []
    base_dir = Path(__file__).parent.parent.parent
    
    # Check phase3_outputs/final_export directory
    phase3_dir = base_dir / "phase3_outputs" / "final_export"
    if phase3_dir.exists():
        for pt_file in phase3_dir.glob("*.pt"):
            try:
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                models.append({
                    "name": pt_file.name,
                    "path": str(pt_file.absolute()),
                    "size_mb": round(size_mb, 2),
                    "location": "phase3_outputs/final_export",
                    "is_current": str(pt_file.absolute()) == _CURRENT_MODEL_PATH,
                    "type": "torchscript"
                })
            except Exception:
                pass
    
    # Check backend/models directory
    models_dir = base_dir / "backend" / "models"
    if models_dir.exists():
        for pt_file in models_dir.glob("*.pt"):
            try:
                size_mb = pt_file.stat().st_size / (1024 * 1024)
                models.append({
                    "name": pt_file.name,
                    "path": str(pt_file.absolute()),
                    "size_mb": round(size_mb, 2),
                    "location": "backend/models",
                    "is_current": str(pt_file.absolute()) == _CURRENT_MODEL_PATH,
                    "type": "torchscript"
                })
            except Exception:
                pass
    
    return models


def get_current_model_info():
    """Get information about the currently loaded model"""
    global _MODEL_LOADED, _CURRENT_MODEL_NAME, _CURRENT_MODEL_PATH, _CURRENT_DEVICE, _MODEL_LOAD_TIME, _SERVER_START_TIME
    
    uptime_seconds = time.time() - _SERVER_START_TIME
    
    return {
        "loaded": _MODEL_LOADED,
        "name": _CURRENT_MODEL_NAME,
        "path": _CURRENT_MODEL_PATH,
        "device": str(_CURRENT_DEVICE) if _CURRENT_DEVICE else "unknown",
        "dummy_mode": not _MODEL_LOADED,
        "model_type": "torchscript",
        "load_time_seconds": _MODEL_LOAD_TIME,
        "uptime_seconds": round(uptime_seconds, 2)
    }


def switch_model(model_path: str):
    """
    Switch to a different TorchScript model
    Args:
        model_path: absolute path to the new model .pt file
    Returns:
        dict with success status and message
    """
    global _MODEL_LOADED, _CURRENT_MODEL, _CURRENT_DEVICE, _CURRENT_MODEL_PATH, _CURRENT_MODEL_NAME, _MODEL_LOAD_TIME
    
    # Validate path exists
    if not os.path.exists(model_path):
        return {
            "success": False,
            "message": f"Model file not found: {model_path}"
        }
    
    # Clear current model
    old_model_name = _CURRENT_MODEL_NAME
    clear_loaded_model()
    
    _CURRENT_MODEL_PATH = model_path
    _CURRENT_MODEL_NAME = Path(model_path).name
    
    # Try to load new model
    try:
        model, device, classes = get_model()
        
        if model is None:
            return {
                "success": False,
                "message": f"Failed to load model from {model_path}",
                "old_model": old_model_name,
                "new_model": _CURRENT_MODEL_NAME
            }
        
        return {
            "success": True,
            "message": f"Successfully switched from {old_model_name} to {_CURRENT_MODEL_NAME}",
            "old_model": old_model_name,
            "new_model": _CURRENT_MODEL_NAME,
            "device": str(device)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error switching model: {str(e)}",
            "old_model": old_model_name
        }


def clear_loaded_model():
    """Force unload the current model (for memory management)"""
    global _MODEL_LOADED, _CURRENT_MODEL
    
    if _CURRENT_MODEL is not None:
        del _CURRENT_MODEL
        _CURRENT_MODEL = None
    
    _MODEL_LOADED = False
    
    # Force garbage collection
    import gc
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {"success": True, "message": "Model cleared from memory"}
