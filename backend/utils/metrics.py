"""
PHASE 5 TIER-0: Metrics Collection
System metrics, GPU metrics, and inference statistics
"""

import psutil
import torch
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
import threading

# Thread-safe metrics storage
_metrics_lock = threading.Lock()
_inference_times: List[float] = []
_emotion_counts: Dict[str, int] = defaultdict(int)


class MetricsCollector:
    """
    Collects and aggregates system and inference metrics
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.start_time = datetime.utcnow()
        self.inference_times: List[float] = []
        self.emotion_counts: Dict[str, int] = defaultdict(int)
        self.lock = threading.Lock()
    
    def record_prediction(self, emotion: str, inference_time_ms: float):
        """
        Record a prediction for metrics
        
        Args:
            emotion: Predicted emotion
            inference_time_ms: Inference time in milliseconds
        """
        with self.lock:
            self.inference_times.append(inference_time_ms)
            self.emotion_counts[emotion] += 1
            
            # Keep only last 10000 predictions to prevent memory bloat
            if len(self.inference_times) > 10000:
                self.inference_times = self.inference_times[-10000:]
    
    def get_system_metrics(self) -> Dict:
        """
        Get current system resource metrics
        
        Returns:
            Dictionary with CPU, memory, and disk metrics
        """
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_total_mb": memory.total / (1024 * 1024),
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_total_gb": disk.total / (1024 * 1024 * 1024)
        }
    
    def get_gpu_metrics(self) -> Dict:
        """
        Get current GPU metrics
        
        Returns:
            Dictionary with GPU availability and metrics
        """
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_name": None,
                "memory_allocated_mb": None,
                "memory_reserved_mb": None,
                "memory_total_mb": None,
                "utilization_percent": None
            }
        
        try:
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(device),
                "memory_allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),
                "memory_reserved_mb": torch.cuda.memory_reserved(device) / (1024 * 1024),
                "memory_total_mb": props.total_memory / (1024 * 1024),
                "utilization_percent": None  # Would need nvidia-ml-py for this
            }
        except Exception:
            return {
                "available": True,
                "device_name": "Unknown",
                "memory_allocated_mb": None,
                "memory_reserved_mb": None,
                "memory_total_mb": None,
                "utilization_percent": None
            }
    
    def get_inference_metrics(self) -> Dict:
        """
        Get inference statistics
        
        Returns:
            Dictionary with inference metrics
        """
        with self.lock:
            if not self.inference_times:
                return {
                    "total_predictions": 0,
                    "avg_inference_time_ms": 0.0,
                    "min_inference_time_ms": 0.0,
                    "max_inference_time_ms": 0.0,
                    "predictions_per_emotion": {},
                    "most_common_emotion": "none"
                }
            
            total_predictions = len(self.inference_times)
            avg_time = sum(self.inference_times) / total_predictions
            min_time = min(self.inference_times)
            max_time = max(self.inference_times)
            
            # Find most common emotion
            if self.emotion_counts:
                most_common = max(self.emotion_counts.items(), key=lambda x: x[1])
                most_common_emotion = most_common[0]
            else:
                most_common_emotion = "none"
            
            return {
                "total_predictions": total_predictions,
                "avg_inference_time_ms": round(avg_time, 2),
                "min_inference_time_ms": round(min_time, 2),
                "max_inference_time_ms": round(max_time, 2),
                "predictions_per_emotion": dict(self.emotion_counts),
                "most_common_emotion": most_common_emotion
            }
    
    def get_uptime_seconds(self) -> float:
        """
        Get server uptime in seconds
        
        Returns:
            Uptime in seconds
        """
        return (datetime.utcnow() - self.start_time).total_seconds()
    
    def get_all_metrics(self) -> Dict:
        """
        Get all metrics combined
        
        Returns:
            Dictionary with system, GPU, and inference metrics
        """
        return {
            "system": self.get_system_metrics(),
            "gpu": self.get_gpu_metrics(),
            "inference": self.get_inference_metrics(),
            "uptime_seconds": round(self.get_uptime_seconds(), 2),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    def reset_metrics(self):
        """Reset all collected metrics"""
        with self.lock:
            self.inference_times.clear()
            self.emotion_counts.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()

