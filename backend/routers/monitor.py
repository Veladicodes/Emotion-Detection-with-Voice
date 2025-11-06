"""
PHASE 5 TIER-0: Enhanced Monitoring & Status Endpoints
Provides system status, GPU info, and detailed metrics with:
- Real-time system resource monitoring
- GPU utilization metrics
- Inference statistics
- Health checks
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import torch
import time
import os
import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# PHASE 5 TIER-0: Import new utilities
try:
    from config import settings
    API_KEY = settings.API_TOKEN
except ImportError:
    API_KEY = "dev-token"

from models.schemas import MetricsResponse, HealthResponse, get_current_timestamp
from utils.model_loader import (
    get_current_model_info,
    get_available_models,
    CLASSES,
    _MODEL_LOADED
)
from utils.metrics import metrics_collector
from utils.security import verify_token, security_scheme

router = APIRouter()
security = HTTPBearer()

# Legacy metrics tracking (kept for backward compatibility)
_TOTAL_REQUESTS = 0
_REQUEST_START_TIME = time.time()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for protected endpoints"""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials


@router.get("/status")
async def get_status():
    """
    Get backend status and uptime
    Public endpoint - no authentication required
    """
    model_info = get_current_model_info()
    uptime = time.time() - _REQUEST_START_TIME
    
    return {
        "status": "online",
        "uptime_seconds": round(uptime, 2),
        "uptime_formatted": format_uptime(uptime),
        "total_requests": _TOTAL_REQUESTS,
        "model": {
            "name": model_info["name"],
            "loaded": model_info["loaded"],
            "device": model_info["device"],
            "type": model_info.get("model_type", "unknown")
        },
        "classes": CLASSES,
        "num_classes": len(CLASSES)
    }


@router.get("/gpu")
async def get_gpu_status():
    """
    Get GPU status and memory information
    Public endpoint
    """
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "message": "CUDA not available. Running on CPU.",
            "device_count": 0
        }
    
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(current_device)
    
    # Memory info (in MB)
    mem_allocated = torch.cuda.memory_allocated(current_device) / (1024 ** 2)
    mem_reserved = torch.cuda.memory_reserved(current_device) / (1024 ** 2)
    mem_total = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 2)
    
    return {
        "cuda_available": True,
        "device_count": device_count,
        "current_device": current_device,
        "gpu_name": gpu_name,
        "memory_allocated_mb": round(mem_allocated, 2),
        "memory_reserved_mb": round(mem_reserved, 2),
        "memory_total_mb": round(mem_total, 2),
        "memory_free_mb": round(mem_total - mem_reserved, 2),
        "utilization_percent": round((mem_reserved / mem_total) * 100, 2) if mem_total > 0 else 0
    }


@router.get("/metrics")
async def get_metrics():
    """
    Get inference metrics and statistics
    Reads from logs/requests.csv
    Public endpoint
    """
    log_file = Path(__file__).parent.parent / "logs" / "requests.csv"
    
    if not log_file.exists():
        return {
            "total_inferences": 0,
            "message": "No inference logs yet"
        }
    
    # Read CSV logs
    emotions_count = {}
    latencies = []
    total = 0
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                emotion = row.get('emotion', 'unknown')
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1
                
                try:
                    latency = float(row.get('latency_ms', 0))
                    latencies.append(latency)
                except:
                    pass
    except Exception as e:
        return {
            "error": f"Failed to read logs: {str(e)}"
        }
    
    # Calculate statistics
    most_frequent = max(emotions_count.items(), key=lambda x: x[1]) if emotions_count else ("none", 0)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    model_info = get_current_model_info()
    
    return {
        "total_inferences": total,
        "emotions_distribution": emotions_count,
        "most_frequent_emotion": {
            "emotion": most_frequent[0],
            "count": most_frequent[1],
            "percentage": round((most_frequent[1] / total * 100), 2) if total > 0 else 0
        },
        "latency_stats": {
            "mean_ms": round(avg_latency, 2),
            "min_ms": round(min_latency, 2),
            "max_ms": round(max_latency, 2)
        },
        "model_info": {
            "name": model_info["name"],
            "device": model_info["device"],
            "uptime_seconds": model_info["uptime_seconds"]
        }
    }


@router.get("/health")
async def health_check():
    """
    Simple health check endpoint
    Returns 200 if server is running
    """
    return {"status": "healthy"}


# PHASE 5 TIER-0: Enhanced metrics endpoint
@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    PHASE 5 TIER-0: Get comprehensive system and inference metrics
    
    Provides real-time information about:
    - System resources (CPU, RAM, Disk)
    - GPU metrics (if available)
    - Inference statistics (total predictions, avg latency, emotion distribution)
    - Server uptime
    
    Returns:
        MetricsResponse: Structured metrics data
    """
    # Get all metrics from the collector
    all_metrics = metrics_collector.get_all_metrics()
    
    # Convert to MetricsResponse model
    from models.schemas import SystemMetrics, GPUMetrics, InferenceMetrics
    
    response = MetricsResponse(
        system=SystemMetrics(**all_metrics["system"]),
        gpu=GPUMetrics(**all_metrics["gpu"]),
        inference=InferenceMetrics(**all_metrics["inference"]),
        uptime_seconds=all_metrics["uptime_seconds"],
        timestamp=all_metrics["timestamp"]
    )
    
    return response


# PHASE 5 TIER-0: Detailed health endpoint
@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """
    PHASE 5 TIER-0: Detailed health check with system status
    
    Provides comprehensive health information:
    - Overall health status
    - Model loading status
    - Device information
    - API version
    - Uptime
    - Warnings (if any)
    
    Returns:
        HealthResponse: Detailed health status
    """
    model_info = get_current_model_info()
    uptime = metrics_collector.get_uptime_seconds()
    
    # Determine health status
    warnings = []
    
    if not model_info["loaded"]:
        warnings.append("Model not loaded - running in dummy mode")
        status = "degraded"
    elif not torch.cuda.is_available():
        warnings.append("GPU not available - using CPU (slower inference)")
        status = "healthy"
    else:
        status = "healthy"
    
    # Build response
    response = HealthResponse(
        status=status,
        uptime_seconds=round(uptime, 2),
        model_loaded=model_info["loaded"],
        model_name=model_info["name"],
        device=model_info["device"],
        version="4.0.0-tier0",
        timestamp=get_current_timestamp(),
        warnings=warnings if warnings else None
    )
    
    return response


def format_uptime(seconds):
    """Format uptime in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def increment_request_counter():
    """Increment global request counter"""
    global _TOTAL_REQUESTS
    _TOTAL_REQUESTS += 1

