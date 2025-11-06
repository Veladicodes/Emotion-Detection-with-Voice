"""
PHASE 5 TIER-0: Pydantic Response Models
Structured data models for API responses and errors
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# ============================================================================
# Response Models
# ============================================================================

class EmotionResponse(BaseModel):
    """
    Standard response for emotion prediction
    
    Example:
        {
            "emotion": "happy",
            "confidence": 0.934,
            "raw_scores": [0.02, 0.01, 0.03, 0.934, 0.01, 0.005, 0.005],
            "inference_time_ms": 213.45,
            "audio_duration_seconds": 3.2,
            "model_used": "EfficientNetV2M+VGG16+DenseNet121",
            "device": "cuda:0",
            "timestamp": "2025-01-06T12:34:56.789Z"
        }
    """
    emotion: str = Field(..., description="Predicted emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    raw_scores: List[float] = Field(..., description="Raw probability scores for all emotions")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    audio_duration_seconds: float = Field(..., description="Audio file duration in seconds")
    model_used: str = Field(..., description="Model architecture used for inference")
    device: str = Field(..., description="Compute device (cuda:0 or cpu)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    warning: Optional[str] = Field(None, description="Warning message if in dummy mode")
    
    class Config:
        json_schema_extra = {
            "example": {
                "emotion": "happy",
                "confidence": 0.934,
                "raw_scores": [0.02, 0.01, 0.03, 0.934, 0.01, 0.005, 0.005],
                "inference_time_ms": 213.45,
                "audio_duration_seconds": 3.2,
                "model_used": "EfficientNetV2M+VGG16+DenseNet121",
                "device": "cuda:0",
                "timestamp": "2025-01-06T12:34:56.789Z"
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response
    
    Example:
        {
            "error": "ValidationError",
            "message": "Unsupported file format",
            "detail": "File must be WAV, MP3, FLAC, OGG, or WebM",
            "timestamp": "2025-01-06T12:34:56.789Z"
        }
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Unsupported file format",
                "detail": "File must be WAV, MP3, FLAC, OGG, or WebM",
                "timestamp": "2025-01-06T12:34:56.789Z"
            }
        }


# ============================================================================
# Monitoring Models
# ============================================================================

class SystemMetrics(BaseModel):
    """System resource metrics"""
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="RAM usage percentage")
    memory_used_mb: float = Field(..., description="RAM used in MB")
    memory_total_mb: float = Field(..., description="Total RAM in MB")
    disk_percent: float = Field(..., description="Disk usage percentage")
    disk_used_gb: float = Field(..., description="Disk used in GB")
    disk_total_gb: float = Field(..., description="Total disk in GB")


class GPUMetrics(BaseModel):
    """GPU resource metrics"""
    available: bool = Field(..., description="GPU availability")
    device_name: Optional[str] = Field(None, description="GPU device name")
    memory_allocated_mb: Optional[float] = Field(None, description="GPU memory allocated in MB")
    memory_reserved_mb: Optional[float] = Field(None, description="GPU memory reserved in MB")
    memory_total_mb: Optional[float] = Field(None, description="Total GPU memory in MB")
    utilization_percent: Optional[float] = Field(None, description="GPU utilization percentage")


class InferenceMetrics(BaseModel):
    """Inference statistics"""
    total_predictions: int = Field(..., description="Total number of predictions")
    avg_inference_time_ms: float = Field(..., description="Average inference time in ms")
    min_inference_time_ms: float = Field(..., description="Minimum inference time in ms")
    max_inference_time_ms: float = Field(..., description="Maximum inference time in ms")
    predictions_per_emotion: Dict[str, int] = Field(..., description="Count per emotion")
    most_common_emotion: str = Field(..., description="Most frequently predicted emotion")


class MetricsResponse(BaseModel):
    """
    Complete metrics response
    
    Example:
        {
            "system": {...},
            "gpu": {...},
            "inference": {...},
            "uptime_seconds": 3600.5,
            "timestamp": "2025-01-06T12:34:56.789Z"
        }
    """
    system: SystemMetrics
    gpu: GPUMetrics
    inference: InferenceMetrics
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class HealthResponse(BaseModel):
    """
    Health check response
    
    Example:
        {
            "status": "healthy",
            "uptime_seconds": 3600.5,
            "model_loaded": true,
            "model_name": "EfficientNetV2M+VGG16+DenseNet121",
            "device": "cuda:0",
            "version": "4.0.0",
            "timestamp": "2025-01-06T12:34:56.789Z"
        }
    """
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    model_loaded: bool = Field(..., description="Model loading status")
    model_name: str = Field(..., description="Loaded model name")
    device: str = Field(..., description="Compute device")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    warnings: Optional[List[str]] = Field(None, description="Health warnings")


# ============================================================================
# Request Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Request model for batch predictions (future use)"""
    audio_files: List[str] = Field(..., description="List of audio file paths or URLs")
    return_raw_scores: bool = Field(default=True, description="Include raw probability scores")


# ============================================================================
# Helper Functions
# ============================================================================

def get_current_timestamp() -> str:
    """Get current timestamp in ISO 8601 format"""
    return datetime.utcnow().isoformat() + "Z"


def create_error_response(error_type: str, message: str, detail: Optional[str] = None) -> ErrorResponse:
    """Create standardized error response"""
    return ErrorResponse(
        error=error_type,
        message=message,
        detail=detail,
        timestamp=get_current_timestamp()
    )

