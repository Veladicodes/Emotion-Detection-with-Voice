"""
PHASE 5 TIER-0: Enhanced Inference Endpoint
Handles emotion prediction from uploaded audio files with:
- Structured response models
- JSON logging
- Security (auth + rate limiting)
- Performance optimizations
Supports WAV, MP3, FLAC, OGG, and WebM (browser recordings)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from fastapi.security import HTTPAuthorizationCredentials
import tempfile
import os
import time
import soundfile as sf
import torch
from datetime import datetime
from typing import Optional

# PHASE 5 TIER-0: Import new utilities
try:
    from config import settings
except ImportError:
    # Fallback for backward compatibility
    class Settings:
        ENABLE_AUTH = False
        API_TOKEN = "dev-token"
        MAX_FILE_SIZE_MB = 50
    settings = Settings()

# PHASE 5 TIER-0: Optional imports with graceful fallbacks
try:
    from models.schemas import EmotionResponse, ErrorResponse, get_current_timestamp
except ImportError:
    EmotionResponse = None
    ErrorResponse = None
    def get_current_timestamp():
        return datetime.utcnow().isoformat() + "Z"

from utils.model_loader import get_model, predict_emotion, _MODEL_LOADED, _CURRENT_MODEL_NAME
from utils.audio_preprocess import preprocess_audio_tensor

# PHASE 5 TIER-0: Import tracer for inference debugging
try:
    from utils.inference_tracer import get_tracer
    TRACING_ENABLED = True
except ImportError:
    get_tracer = lambda: None
    TRACING_ENABLED = False
    print("Warning: Inference tracer not available")

try:
    from utils.logging_utils import prediction_logger
except ImportError:
    prediction_logger = None

try:
    from utils.security import verify_token, security_scheme, limiter, get_client_ip
except ImportError:
    verify_token = lambda *args, **kwargs: True
    security_scheme = None
    limiter = None
    def get_client_ip(request):
        return request.client.host if request.client else "unknown"

try:
    from utils.metrics import metrics_collector
except ImportError:
    class DummyMetricsCollector:
        def record_prediction(self, *args, **kwargs):
            pass
    metrics_collector = DummyMetricsCollector()

router = APIRouter()


def convert_webm_to_wav(webm_path: str) -> str:
    """
    Convert WebM to WAV using pydub
    Returns path to converted WAV file
    """
    try:
        from pydub import AudioSegment
        
        # Load WebM
        audio = AudioSegment.from_file(webm_path, format="webm")
        
        # Convert to WAV
        wav_path = webm_path.replace('.webm', '_converted.wav')
        audio.export(wav_path, format="wav")
        
        return wav_path
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="pydub not installed. Install with: pip install pydub"
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to convert WebM to WAV: {str(e)}"
        )

# Model is lazily loaded on first request (so server starts quickly)
MODEL, DEVICE, CLASSES = None, None, None

def ensure_model():
    """Ensure model is loaded before inference"""
    global MODEL, DEVICE, CLASSES
    if MODEL is None:
        MODEL, DEVICE, CLASSES = get_model()
    return MODEL, DEVICE, CLASSES


# PHASE 5 TIER-0: Enhanced prediction endpoint with security, logging, and metrics
# Apply rate limiting decorator conditionally
if limiter:
    @router.post("/predict", response_model=EmotionResponse if EmotionResponse else None)
    @limiter.limit("100/minute")
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme if security_scheme else None)
    ):
        return await _predict_handler(request, file, credentials)
else:
    @router.post("/predict", response_model=EmotionResponse if EmotionResponse else None)
    async def predict(
        request: Request,
        file: UploadFile = File(...),
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme if security_scheme else None)
    ):
        return await _predict_handler(request, file, credentials)


async def _predict_handler(
    request: Request,
    file: UploadFile,
    credentials: Optional[HTTPAuthorizationCredentials]
):
    """
    PHASE 5 TIER-0: Enhanced emotion prediction endpoint
    
    Predict emotion from uploaded audio file with:
    - Structured response (EmotionResponse model)
    - JSON logging
    - Bearer token authentication (if enabled)
    - Rate limiting (100 requests/minute per IP)
    - Metrics collection
    
    Args:
        request: FastAPI request object (for IP extraction)
        file: Audio file (wav, mp3, flac, ogg, webm)
        credentials: Optional Bearer token credentials
    
    Returns:
        EmotionResponse: Structured response with emotion, confidence, timing, etc.
    
    Raises:
        HTTPException: On validation, authentication, or processing errors
    """
    # PHASE 5 TIER-0: Verify authentication if enabled
    try:
        verify_token(credentials, settings.API_TOKEN, settings.ENABLE_AUTH)
    except HTTPException:
        # Log failed auth attempt
        if prediction_logger:
            prediction_logger.log_error(
                filename=file.filename,
                error_type="AuthenticationError",
                error_message="Invalid or missing token",
                client_ip=get_client_ip(request)
            )
        raise
    
    # Get client IP for logging
    client_ip = get_client_ip(request)
    
    # PHASE 5 TIER-0: Validate file format
    allowed_formats = (".wav", ".flac", ".mp3", ".ogg", ".webm")
    if not file.filename.lower().endswith(allowed_formats):
        error_msg = f"Unsupported file format. Use wav/flac/mp3/ogg/webm. Got: {file.filename}"
        
        # Log error
        if prediction_logger:
            prediction_logger.log_error(
                filename=file.filename,
                error_type="ValidationError",
                error_message=error_msg,
                client_ip=client_ip
            )
        
        raise HTTPException(status_code=400, detail=error_msg)
    
    # PHASE 5 TIER-0: Check file size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        error_msg = f"File too large: {file_size_mb:.2f}MB. Maximum: {settings.MAX_FILE_SIZE_MB}MB"
        
        if prediction_logger:
            prediction_logger.log_error(
                filename=file.filename,
                error_type="FileSizeError",
                error_message=error_msg,
                client_ip=client_ip
            )
        
        raise HTTPException(status_code=413, detail=error_msg)
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    
    # PHASE 5 TIER-0: Start precise timing for inference
    start_time = time.perf_counter()
    
    # Handle WebM format (convert to WAV first)
    is_webm = file.filename.lower().endswith('.webm')
    audio_path = tmp_path
    
    if is_webm:
        try:
            audio_path = convert_webm_to_wav(tmp_path)
        except Exception as e:
            os.remove(tmp_path)
            
            if prediction_logger:
                prediction_logger.log_error(
                    filename=file.filename,
                    error_type="ConversionError",
                    error_message=str(e),
                    client_ip=client_ip
                )
            
            raise HTTPException(
                status_code=400,
                detail=f"Failed to convert WebM: {str(e)}. Ensure ffmpeg is installed."
            )
    
    # PHASE 5 TIER-0: Start inference tracing
    tracer = get_tracer() if TRACING_ENABLED else None
    if tracer:
        tracer.start_trace(file.filename)
    
    try:
        # Read audio file
        wav, sr = sf.read(audio_path, dtype='float32')
        audio_duration = len(wav) / sr
        
        # Ensure model is loaded
        model, device, classes = ensure_model()
        
        # ADAPTIVE PREPROCESSING: Detect if this is a live recording
        # Live recordings: filename starts with "recording" or webm/ogg format
        is_live = (
            file.filename.lower().startswith("recording") or 
            file.content_type in ["audio/ogg", "audio/webm", "audio/ogg; codecs=opus", "audio/webm; codecs=opus"]
        )
        
        # PHASE 5 TIER-0: Preprocess with tracing and adaptive mode
        mel_tensor = preprocess_audio_tensor(wav, sr, device=device, tracer=tracer, is_live=is_live)
        
        # PHASE 5 TIER-0: Inference with tracing
        pred_label, conf, raw = predict_emotion(model, mel_tensor, device, classes, tracer=tracer)
        
        # PHASE 5 TIER-0: Calculate precise inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # PHASE 5 TIER-0: Record metrics
        metrics_collector.record_prediction(pred_label, inference_time_ms)
        
        # PHASE 5 TIER-0: Log prediction to JSON
        if prediction_logger:
            prediction_logger.log_prediction(
                filename=file.filename,
                emotion=pred_label,
                confidence=conf,
                inference_time_ms=inference_time_ms,
                audio_duration=audio_duration,
                model_name=_CURRENT_MODEL_NAME,
                device=str(device),
                client_ip=client_ip,
                raw_scores=raw
            )
        
        # PHASE 5 TIER-0: Build structured response
        if EmotionResponse is not None:
            response = EmotionResponse(
                emotion=pred_label,
                confidence=float(conf),
                raw_scores=raw,
                inference_time_ms=round(inference_time_ms, 2),
                audio_duration_seconds=round(audio_duration, 2),
                model_used=_CURRENT_MODEL_NAME,
                device=str(device),
                timestamp=get_current_timestamp(),
                warning="WARNING: DUMMY MODE - Returning random predictions" if not _MODEL_LOADED else None
            )
        else:
            # Fallback to dict response for backward compatibility
            response = {
                "emotion": pred_label,
                "confidence": float(conf),
                "raw_scores": raw,
                "inference_time_ms": round(inference_time_ms, 2),
                "audio_duration_seconds": round(audio_duration, 2),
                "model": _CURRENT_MODEL_NAME,
                "device": str(device),
                "timestamp": get_current_timestamp()
            }
            if not _MODEL_LOADED:
                response["warning"] = "WARNING: DUMMY MODE - Returning random predictions"
        
        # PHASE 5 TIER-0: End trace
        if tracer:
            tracer.end_trace()
        
        return response
    
    except HTTPException as http_exc:
        if tracer:
            tracer.log_error("HTTP_EXCEPTION", str(http_exc))
            tracer.end_trace()
        raise
    except Exception as e:
        # Log unexpected error
        if prediction_logger:
            prediction_logger.log_error(
                filename=file.filename,
                error_type="InferenceError",
                error_message=str(e),
                client_ip=client_ip
            )
        
        if tracer:
            tracer.log_error("INFERENCE_PIPELINE", str(e), {"exception_type": type(e).__name__})
            tracer.end_trace()
        
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    finally:
        # Cleanup temporary files
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if is_webm and audio_path != tmp_path and os.path.exists(audio_path):
            os.remove(audio_path)


def log_prediction(filename: str, emotion: str, confidence: float, latency_ms: float, model_name: str, gpu_used: bool):
    """
    Log prediction to CSV file
    Creates file with headers if it doesn't exist
    """
    LOG_CSV = os.path.join(os.path.dirname(__file__), "..", "logs", "requests.csv")
    os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)
    
    # Create CSV with headers if doesn't exist
    file_exists = os.path.exists(LOG_CSV)
    
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        
        # Write headers if new file
        if not file_exists:
            writer.writerow([
                "timestamp",
                "filename",
                "emotion",
                "confidence",
                "latency_ms",
                "model",
                "gpu_used"
            ])
        
        # Write prediction data
        writer.writerow([
            time.time(),
            filename,
            emotion,
            float(confidence),
            latency_ms,
            model_name,
            gpu_used
        ])

