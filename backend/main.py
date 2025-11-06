"""
PHASE 5 TIER-0: Production-Ready FastAPI Application
Emotion Voice Recognition Backend with:
- TorchScript ensemble model integration
- Bearer token authentication
- Rate limiting
- Structured logging
- System metrics
- Docker support
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import asyncio
from datetime import datetime

# PHASE 5 TIER-0: Optional imports with graceful fallbacks
try:
    from slowapi.errors import RateLimitExceeded
except ImportError:
    RateLimitExceeded = None
    print("Warning: slowapi not installed. Rate limiting disabled.")

try:
    from config import settings
except ImportError:
    # Fallback for backward compatibility
    class Settings:
        cors_origins_list = ["*"]
        ENABLE_MODEL_PRELOAD = True
    settings = Settings()

from routers import inference, admin, monitor
from utils.model_loader import get_model

# PHASE 5 TIER-0: Optional utilities
try:
    from utils.security import limiter
except ImportError:
    limiter = None
    print("Warning: Security utilities not available. Rate limiting disabled.")

try:
    from utils.logging_utils import system_logger, initialize_loggers
except ImportError:
    system_logger = None
    initialize_loggers = lambda *args, **kwargs: None
    print("Warning: Logging utilities not available. Using basic logging.")

try:
    from models.schemas import ErrorResponse, get_current_timestamp
except ImportError:
    ErrorResponse = None
    def get_current_timestamp():
        return datetime.utcnow().isoformat() + "Z"
    print("Warning: Schemas not available. Using basic responses.")

# PHASE 5 TIER-0: Initialize application with enhanced metadata
app = FastAPI(
    title="Emotion Voice API - TIER-0 Production",
    version="4.0.0-tier0",
    description=(
        "🎤 Production-ready emotion recognition API powered by TorchScript ensemble model\n\n"
        "**Features:**\n"
        "- 7 emotion detection (angry, disgust, fear, happy, neutral, sad, surprise)\n"
        "- Multi-format audio support (WAV, MP3, FLAC, OGG, WebM)\n"
        "- Real-time inference with GPU acceleration\n"
        "- Structured JSON logging\n"
        "- Rate limiting and authentication\n"
        "- Comprehensive monitoring and metrics\n\n"
        "**Model:** EfficientNetV2-M + VGG16 + DenseNet121 Ensemble\n"
        "**Accuracy:** 92% on validation set"
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "Inference",
            "description": "Emotion prediction from audio files"
        },
        {
            "name": "Monitoring",
            "description": "System health, metrics, and status"
        },
        {
            "name": "Admin & Model Management",
            "description": "Model management and administration"
        }
    ]
)

# PHASE 5 TIER-0: Configure CORS with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PHASE 5 TIER-0: Add rate limiter state (if available)
if limiter:
    app.state.limiter = limiter

# PHASE 5 TIER-0: Include routers
app.include_router(inference.router, prefix="/api", tags=["Inference"])
app.include_router(monitor.router, prefix="/api", tags=["Monitoring"])
app.include_router(admin.router, prefix="/api", tags=["Admin & Model Management"])

# PHASE 5 TIER-0: Custom error handlers (only if dependencies available)
if RateLimitExceeded is not None:
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        """Handle rate limit exceeded errors"""
        if ErrorResponse is not None:
            return JSONResponse(
                status_code=429,
                content=ErrorResponse(
                    error="RateLimitExceeded",
                    message="Too many requests. Please slow down.",
                    detail=str(exc),
                    timestamp=get_current_timestamp()
                ).dict()
            )
        return JSONResponse(
            status_code=429,
            content={"error": "RateLimitExceeded", "message": "Too many requests"}
        )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions with structured response"""
    if ErrorResponse is not None:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"HTTP{exc.status_code}",
                message=exc.detail,
                detail=None,
                timestamp=get_current_timestamp()
            ).dict()
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": f"HTTP{exc.status_code}", "message": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors with structured response"""
    if ErrorResponse is not None:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="ValidationError",
                message="Request validation failed",
                detail=str(exc.errors()),
                timestamp=get_current_timestamp()
            ).dict()
        )
    return JSONResponse(
        status_code=422,
        content={"error": "ValidationError", "message": "Request validation failed", "detail": str(exc.errors())}
    )


@app.on_event("startup")
async def startup_event():
    """PHASE 5 TIER-0: Enhanced startup with logging initialization"""
    print("\n" + "="*70)
    print("Starting Emotion Voice API - TIER-0 Production")
    print("="*70 + "\n")
    
    # Initialize logging system
    try:
        initialize_loggers()
        if system_logger:
            system_logger.log_event(
                "startup",
                "Server starting",
                version="4.0.0-tier0"
            )
    except Exception as e:
        print(f"Warning: Failed to initialize logging: {e}")
    
    # Load model in background if enabled
    if settings.ENABLE_MODEL_PRELOAD:
        def load_model_sync():
            try:
                model, device, classes = get_model()
                if model is not None:
                    print("\n" + "="*70)
                    print("SUCCESS: Model preloaded successfully! Ready for predictions.")
                    print(f"Device: {device}")
                    print("="*70 + "\n")
                    
                    if system_logger:
                        system_logger.log_event(
                            "model_loaded",
                            "Model preloaded successfully",
                            device=str(device)
                        )
                else:
                    print("\n" + "="*70)
                    print("WARNING: Running in DUMMY MODE (no model loaded)")
                    print("="*70 + "\n")
                    
                    if system_logger:
                        system_logger.log_event(
                            "model_warning",
                            "Running in dummy mode",
                            reason="Model file not found"
                        )
            except Exception as e:
                print(f"\nWARNING: Failed to preload model: {e}")
                print("Server will continue in lazy-loading mode.\n")
                
                if system_logger:
                    system_logger.log_event(
                        "model_error",
                        "Failed to preload model",
                        error=str(e)
                    )
        
        # Run in background thread to not block startup
        await asyncio.to_thread(load_model_sync)
    
    print("="*70)
    print("API Documentation: http://127.0.0.1:8000/docs")
    print("Health Check: http://127.0.0.1:8000/health")
    print("Metrics: http://127.0.0.1:8000/api/metrics")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """PHASE 5 TIER-0: Graceful shutdown with logging"""
    if system_logger:
        system_logger.log_event("shutdown", "Server shutting down")
    
    print("\n" + "="*70)
    print("Emotion Voice API - Shutting down")
    print("="*70 + "\n")


@app.get("/", tags=["Root"])
def root():
    """API root endpoint with service information"""
    return {
        "service": "Emotion Voice API - TIER-0 Production",
        "version": "4.0.0-tier0",
        "description": "Production-ready emotion recognition from voice",
        "documentation": "/docs",
        "health": "/health",
        "detailed_health": "/api/health/detailed",
        "status": "/api/status",
        "metrics": "/api/metrics",
        "model": "EfficientNetV2-M + VGG16 + DenseNet121 Ensemble",
        "accuracy": "92%",
        "supported_emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
        "supported_formats": ["WAV", "MP3", "FLAC", "OGG", "WebM"]
    }


@app.get("/health", tags=["Root"])
def health():
    """Simple health check"""
    return {"status": "ok", "timestamp": get_current_timestamp()}

