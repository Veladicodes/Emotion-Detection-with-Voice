"""
PHASE 5 TIER-0: Structured JSON Logging
Thread-safe rotating JSON logs for predictions and system events
"""

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading

# Thread-safe lock for logging
_log_lock = threading.Lock()

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_logger(
    name: str,
    log_file: str,
    level: str = "INFO",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with rotating JSON file handler
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create rotating file handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(JSONFormatter())
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


class PredictionLogger:
    """
    Thread-safe logger for emotion predictions
    
    Logs predictions to rotating JSON files with structured data
    """
    
    def __init__(
        self,
        log_dir: str = "logs/predictions",
        max_bytes: int = 10485760,
        backup_count: int = 5
    ):
        """
        Initialize prediction logger
        
        Args:
            log_dir: Directory for prediction logs
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.log_dir / "predictions.log"
        self.logger = setup_logger(
            "predictions",
            str(log_file),
            level="INFO",
            max_bytes=max_bytes,
            backup_count=backup_count
        )
    
    def log_prediction(
        self,
        filename: str,
        emotion: str,
        confidence: float,
        inference_time_ms: float,
        audio_duration: float,
        model_name: str,
        device: str,
        client_ip: Optional[str] = None,
        raw_scores: Optional[list] = None
    ):
        """
        Log an emotion prediction
        
        Args:
            filename: Original audio filename
            emotion: Predicted emotion
            confidence: Prediction confidence
            inference_time_ms: Inference time in milliseconds
            audio_duration: Audio duration in seconds
            model_name: Model used for prediction
            device: Compute device (cuda:0 or cpu)
            client_ip: Client IP address
            raw_scores: Raw probability scores
        """
        with _log_lock:
            extra_data = {
                "event_type": "prediction",
                "filename": filename,
                "emotion": emotion,
                "confidence": round(confidence, 4),
                "inference_time_ms": round(inference_time_ms, 2),
                "audio_duration_seconds": round(audio_duration, 2),
                "model_name": model_name,
                "device": device,
                "client_ip": client_ip or "unknown"
            }
            
            if raw_scores:
                extra_data["raw_scores"] = [round(s, 4) for s in raw_scores]
            
            # Create log record with extra data
            record = self.logger.makeRecord(
                self.logger.name,
                logging.INFO,
                __file__,
                0,
                f"Prediction: {emotion}",
                (),
                None
            )
            record.extra_data = extra_data
            
            self.logger.handle(record)
    
    def log_error(
        self,
        filename: str,
        error_type: str,
        error_message: str,
        client_ip: Optional[str] = None
    ):
        """
        Log a prediction error
        
        Args:
            filename: Original audio filename
            error_type: Type of error
            error_message: Error message
            client_ip: Client IP address
        """
        with _log_lock:
            extra_data = {
                "event_type": "error",
                "filename": filename,
                "error_type": error_type,
                "error_message": error_message,
                "client_ip": client_ip or "unknown"
            }
            
            record = self.logger.makeRecord(
                self.logger.name,
                logging.ERROR,
                __file__,
                0,
                f"Error: {error_type}",
                (),
                None
            )
            record.extra_data = extra_data
            
            self.logger.handle(record)


class SystemLogger:
    """
    Logger for system events (startup, shutdown, errors)
    """
    
    def __init__(
        self,
        log_dir: str = "logs/system",
        max_bytes: int = 10485760,
        backup_count: int = 5
    ):
        """
        Initialize system logger
        
        Args:
            log_dir: Directory for system logs
            max_bytes: Max log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = self.log_dir / "system.log"
        self.logger = setup_logger(
            "system",
            str(log_file),
            level="INFO",
            max_bytes=max_bytes,
            backup_count=backup_count
        )
    
    def log_event(self, event_type: str, message: str, **kwargs):
        """
        Log a system event
        
        Args:
            event_type: Type of event (startup, shutdown, error, etc.)
            message: Event message
            **kwargs: Additional event data
        """
        with _log_lock:
            extra_data = {
                "event_type": event_type,
                **kwargs
            }
            
            record = self.logger.makeRecord(
                self.logger.name,
                logging.INFO,
                __file__,
                0,
                message,
                (),
                None
            )
            record.extra_data = extra_data
            
            self.logger.handle(record)


# ============================================================================
# Global Logger Instances
# ============================================================================

# These will be initialized when the module is imported
prediction_logger: Optional[PredictionLogger] = None
system_logger: Optional[SystemLogger] = None


def initialize_loggers(log_dir: str = "logs", max_bytes: int = 10485760, backup_count: int = 5):
    """
    Initialize global logger instances
    
    Args:
        log_dir: Base directory for logs
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
    """
    global prediction_logger, system_logger
    
    prediction_logger = PredictionLogger(
        log_dir=f"{log_dir}/predictions",
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    
    system_logger = SystemLogger(
        log_dir=f"{log_dir}/system",
        max_bytes=max_bytes,
        backup_count=backup_count
    )
    
    system_logger.log_event("startup", "Logging system initialized")


# Initialize loggers on module import
initialize_loggers()

