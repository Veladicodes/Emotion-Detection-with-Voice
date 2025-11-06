"""
PHASE 5 TIER-0: Centralized Configuration Management
Loads environment variables and provides typed configuration
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    HOST: str = Field(default="127.0.0.1", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    RELOAD: bool = Field(default=True, description="Auto-reload on code changes")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    
    # Security Configuration
    CORS_ORIGINS: str = Field(
        default="http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000",
        description="Comma-separated list of allowed origins"
    )
    API_TOKEN: str = Field(
        default="dev-token-change-in-production",
        description="Bearer token for API authentication"
    )
    ENABLE_AUTH: bool = Field(default=False, description="Enable Bearer token authentication")
    RATE_LIMIT: str = Field(default="100/minute", description="Rate limit per IP")
    
    # Model Configuration
    MODEL_PATH: str = Field(
        default=r"D:\Emotion Voice\phase3_outputs\final_export\best_efficientnet_v2_m+vgg16+densenet121.pt",
        description="Path to TorchScript model"
    )
    DEVICE: str = Field(default="auto", description="Device: auto, cuda, or cpu")
    ENABLE_MODEL_PRELOAD: bool = Field(default=True, description="Preload model at startup")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_DIR: str = Field(default="logs", description="Log directory")
    LOG_MAX_BYTES: int = Field(default=10485760, description="Max log file size (10MB)")
    LOG_BACKUP_COUNT: int = Field(default=5, description="Number of backup log files")
    
    # Performance Configuration
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Maximum upload file size in MB")
    ASYNC_THRESHOLD_MB: int = Field(default=1, description="File size threshold for async processing")
    ENABLE_CACHE: bool = Field(default=True, description="Enable model caching")
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable metrics collection")
    METRICS_RETENTION_DAYS: int = Field(default=30, description="Metrics retention period")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def max_file_size_bytes(self) -> int:
        """Convert max file size to bytes"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    @property
    def async_threshold_bytes(self) -> int:
        """Convert async threshold to bytes"""
        return self.ASYNC_THRESHOLD_MB * 1024 * 1024

# Global settings instance
settings = Settings()

# Ensure log directory exists
Path(settings.LOG_DIR).mkdir(parents=True, exist_ok=True)
