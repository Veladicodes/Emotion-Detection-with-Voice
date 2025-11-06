"""
Admin endpoints for model management, monitoring, and metrics
Protected by API key authentication
"""
from fastapi import APIRouter, HTTPException, Header, Depends
from typing import Optional
import torch
import csv
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

from utils.model_loader import (
    get_available_models,
    get_current_model_info,
    switch_model,
    clear_loaded_model
)
# PHASE 5 TIER-0: Use new configuration system
try:
    from config import settings
    ADMIN_API_KEY = settings.API_TOKEN
except ImportError:
    ADMIN_API_KEY = "dev-token"

import os
LOG_CSV = os.path.join(os.path.dirname(__file__), "..", "logs", "requests.csv")

router = APIRouter()

# ============================================================================
# Authentication Dependency
# ============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify admin API key for protected endpoints"""
    if x_api_key is None:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include 'x-api-key' header."
        )
    if x_api_key != ADMIN_API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return x_api_key

# ============================================================================
# Model Management Endpoints
# ============================================================================

@router.get("/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    """
    List all available model checkpoints
    
    Returns:
        - List of models in training_outputs/ and backend/models/
        - Each entry shows: name, path, size, location, is_current
    """
    models = get_available_models()
    current_model = get_current_model_info()
    
    return {
        "total_models": len(models),
        "current_model": current_model,
        "available_models": models
    }

@router.post("/switch_model")
async def switch_active_model(
    model_name: str,
    api_key: str = Depends(verify_api_key)
):
    """
    Switch to a different model checkpoint
    
    Args:
        model_name: name of the model file (e.g., "best_resnet50_adam_relu.pt")
    
    Returns:
        Success status and new model info
    """
    # Find the model path
    available_models = get_available_models()
    model_path = None
    
    for model in available_models:
        if model["name"] == model_name:
            model_path = model["path"]
            break
    
    if model_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Use /api/models to see available models."
        )
    
    # Switch model
    result = switch_model(model_path)
    
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    
    return result

@router.post("/clear_model")
async def clear_model_memory(api_key: str = Depends(verify_api_key)):
    """
    Clear currently loaded model from memory (force garbage collection)
    Useful for memory management before switching models
    """
    result = clear_loaded_model()
    return result

# ============================================================================
# GPU Health Endpoint
# ============================================================================

@router.get("/gpu")
async def get_gpu_info():
    """
    Get GPU status, memory usage, and CUDA information
    Public endpoint - no API key required
    """
    info = {
        "gpu_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_id = 0
        info.update({
            "device_name": torch.cuda.get_device_name(device_id),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cudnn_enabled": torch.backends.cudnn.enabled,
        })
        
        # Get memory info
        try:
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)  # GB
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB
            
            info.update({
                "memory_allocated_GB": round(memory_allocated, 2),
                "memory_reserved_GB": round(memory_reserved, 2),
                "memory_total_GB": round(memory_total, 2),
                "memory_free_GB": round(memory_total - memory_reserved, 2),
                "memory_utilization_percent": round((memory_reserved / memory_total) * 100, 2)
            })
        except Exception as e:
            info["memory_error"] = str(e)
    else:
        info["message"] = "CUDA not available. Running on CPU."
    
    return info

# ============================================================================
# Metrics Endpoint
# ============================================================================

@router.get("/metrics")
async def get_metrics(api_key: str = Depends(verify_api_key)):
    """
    Get API usage metrics from request logs
    
    Returns:
        - Total requests
        - Average latency
        - Top predicted emotions
        - Current model info
    """
    if not os.path.exists(LOG_CSV):
        return {
            "total_requests": 0,
            "message": "No requests logged yet",
            "current_model": get_current_model_info()
        }
    
    # Read CSV log
    requests_data = []
    emotion_counts = Counter()
    latencies = []
    models_used = Counter()
    
    try:
        with open(LOG_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                requests_data.append(row)
                
                # Count emotions
                if "emotion" in row:
                    emotion_counts[row["emotion"]] += 1
                
                # Collect latencies
                if "latency_ms" in row:
                    try:
                        latencies.append(float(row["latency_ms"]))
                    except (ValueError, TypeError):
                        pass
                
                # Count models used
                if "model" in row:
                    models_used[row["model"]] += 1
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")
    
    # Calculate metrics
    total_requests = len(requests_data)
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    
    # Get top 5 emotions
    top_emotions = dict(emotion_counts.most_common(5))
    
    return {
        "total_requests": total_requests,
        "avg_latency_ms": avg_latency,
        "min_latency_ms": round(min(latencies), 2) if latencies else 0,
        "max_latency_ms": round(max(latencies), 2) if latencies else 0,
        "top_emotions": top_emotions,
        "models_used": dict(models_used),
        "current_model": get_current_model_info(),
        "gpu_info": await get_gpu_info()
    }

# ============================================================================
# System Status Endpoint
# ============================================================================

@router.get("/status")
async def get_system_status():
    """
    Get overall system status (public endpoint)
    """
    current_model = get_current_model_info()
    gpu_info = await get_gpu_info()
    
    # Count requests (simple count without reading full CSV)
    request_count = 0
    if os.path.exists(LOG_CSV):
        try:
            with open(LOG_CSV, "r", encoding="utf-8") as f:
                request_count = sum(1 for _ in f) - 1  # subtract header
        except Exception:
            pass
    
    return {
        "server": "Emotion Voice API",
        "status": "running",
        "model": {
            "loaded": current_model["loaded"],
            "name": current_model["name"],
            "dummy_mode": current_model["dummy_mode"]
        },
        "gpu": {
            "available": gpu_info["gpu_available"],
            "device": gpu_info.get("device_name", "CPU")
        },
        "total_requests": request_count,
        "timestamp": datetime.now().isoformat()
    }

