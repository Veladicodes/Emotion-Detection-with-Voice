"""
PHASE 5 TIER-0: Inference Pipeline Tracer
Detailed logging of every step in the audio → spectrogram → model → prediction flow
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class InferenceTracer:
    """
    Traces and logs every step of the inference pipeline for debugging
    """
    
    def __init__(self, log_dir: str = "logs/inference_trace"):
        """
        Initialize the tracer
        
        Args:
            log_dir: Directory to store trace logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_trace = {}
        self.trace_file = None
    
    def start_trace(self, filename: str):
        """
        Start a new trace for an inference request
        
        Args:
            filename: Name of the audio file being processed
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        self.trace_file = self.log_dir / f"trace_{timestamp}_{filename}.log"
        
        self.current_trace = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "filename": filename,
            "stages": []
        }
        
        self._log("="*70)
        self._log(f"INFERENCE TRACE STARTED: {filename}")
        self._log(f"Timestamp: {self.current_trace['timestamp']}")
        self._log("="*70)
    
    def log_stage(self, stage_name: str, data: Dict[str, Any]):
        """
        Log a stage in the pipeline
        
        Args:
            stage_name: Name of the pipeline stage
            data: Data to log for this stage
        """
        stage_data = {
            "stage": stage_name,
            "data": self._serialize_data(data)
        }
        
        self.current_trace["stages"].append(stage_data)
        
        self._log(f"\n[STAGE] {stage_name}")
        self._log("-" * 70)
        for key, value in data.items():
            self._log(f"  {key}: {value}")
    
    def log_audio_loaded(self, sr: int, duration: float, shape: tuple, dtype: str, 
                        channels: int, min_val: float, max_val: float, mean_val: float):
        """Log audio file loading details"""
        self.log_stage("1. AUDIO_LOADED", {
            "sample_rate": sr,
            "duration_seconds": round(duration, 3),
            "waveform_shape": str(shape),
            "dtype": dtype,
            "num_channels": channels,
            "min_amplitude": round(min_val, 6),
            "max_amplitude": round(max_val, 6),
            "mean_amplitude": round(mean_val, 6),
            "status": "✓ Audio decoded successfully"
        })
    
    def log_resampling(self, original_sr: int, target_sr: int, new_shape: tuple):
        """Log audio resampling"""
        self.log_stage("2. RESAMPLING", {
            "original_sample_rate": original_sr,
            "target_sample_rate": target_sr,
            "new_waveform_shape": str(new_shape),
            "resampled": "Yes" if original_sr != target_sr else "No (already correct)",
            "status": "✓ Resampling complete"
        })
    
    def log_mono_conversion(self, original_channels: int, new_shape: tuple):
        """Log mono conversion"""
        self.log_stage("3. MONO_CONVERSION", {
            "original_channels": original_channels,
            "converted_to": "mono (1 channel)",
            "new_shape": str(new_shape),
            "method": "average across channels",
            "status": "✓ Mono conversion complete"
        })
    
    def log_mel_spectrogram(self, params: Dict[str, Any], mel_shape: tuple, 
                           mel_min: float, mel_max: float, mel_mean: float):
        """Log MelSpectrogram generation"""
        self.log_stage("4. MEL_SPECTROGRAM", {
            "sample_rate": params.get("sample_rate"),
            "n_fft": params.get("n_fft"),
            "hop_length": params.get("hop_length"),
            "n_mels": params.get("n_mels"),
            "mel_shape": str(mel_shape),
            "mel_min": round(mel_min, 6),
            "mel_max": round(mel_max, 6),
            "mel_mean": round(mel_mean, 6),
            "status": "✓ MelSpectrogram generated"
        })
    
    def log_db_conversion(self, db_shape: tuple, db_min: float, db_max: float, db_mean: float):
        """Log dB scale conversion"""
        self.log_stage("5. DB_CONVERSION", {
            "db_shape": str(db_shape),
            "db_min": round(db_min, 6),
            "db_max": round(db_max, 6),
            "db_mean": round(db_mean, 6),
            "transform": "AmplitudeToDB",
            "status": "✓ dB conversion complete"
        })
    
    def log_normalization(self, before_mean: float, before_std: float, 
                         after_mean: float, after_std: float, normalized_shape: tuple):
        """Log normalization (z-score)"""
        self.log_stage("6. NORMALIZATION", {
            "before_mean": round(before_mean, 6),
            "before_std": round(before_std, 6),
            "after_mean": round(after_mean, 6),
            "after_std": round(after_std, 6),
            "normalized_shape": str(normalized_shape),
            "method": "z-score (mean=0, std=1)",
            "formula": "(x - mean) / (std + 1e-6)",
            "status": "✓ Normalization complete"
        })
    
    def log_resize(self, original_shape: tuple, target_size: tuple, final_shape: tuple,
                  method: str = "bilinear"):
        """Log resizing to target dimensions"""
        self.log_stage("7. RESIZE", {
            "original_shape": str(original_shape),
            "target_size": str(target_size),
            "final_shape": str(final_shape),
            "interpolation_method": method,
            "align_corners": False,
            "status": "✓ Resize complete"
        })
    
    def log_model_input(self, input_shape: tuple, input_dtype: str, device: str,
                       has_nan: bool, has_inf: bool, min_val: float, max_val: float):
        """Log model input validation"""
        issues = []
        if has_nan:
            issues.append("❌ NaNs detected")
        if has_inf:
            issues.append("❌ Infs detected")
        
        status = "✓ Input valid" if not issues else " | ".join(issues)
        
        self.log_stage("8. MODEL_INPUT_VALIDATION", {
            "input_shape": str(input_shape),
            "expected_shape": "(1, 1, 128, 128)",
            "shape_match": "✓ Yes" if str(input_shape) == "(1, 1, 128, 128)" else f"❌ No: {input_shape}",
            "dtype": input_dtype,
            "device": device,
            "min_value": round(min_val, 6),
            "max_value": round(max_val, 6),
            "has_nan": has_nan,
            "has_inf": has_inf,
            "status": status
        })
    
    def log_model_inference(self, model_name: str, device: str, logits: np.ndarray,
                           probabilities: np.ndarray, predicted_class: int, 
                           predicted_emotion: str, confidence: float, inference_time_ms: float):
        """Log model inference results"""
        self.log_stage("9. MODEL_INFERENCE", {
            "model_name": model_name,
            "device": device,
            "inference_time_ms": round(inference_time_ms, 2),
            "logits_shape": str(logits.shape),
            "logits": [round(float(x), 6) for x in logits],
            "probabilities": [round(float(x), 6) for x in probabilities],
            "predicted_class_index": int(predicted_class),
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence, 6),
            "status": "✓ Inference complete"
        })
    
    def log_parameter_comparison(self, training_params: Dict[str, Any], 
                                inference_params: Dict[str, Any]):
        """Log comparison between training and inference parameters"""
        mismatches = []
        for key in training_params:
            if key in inference_params:
                if training_params[key] != inference_params[key]:
                    mismatches.append(f"{key}: train={training_params[key]}, inference={inference_params[key]}")
        
        status = "✓ All parameters match" if not mismatches else "⚠️ Parameter mismatches detected"
        
        self.log_stage("PARAMETER_VALIDATION", {
            "training_params": training_params,
            "inference_params": inference_params,
            "mismatches": mismatches if mismatches else "None",
            "status": status
        })
    
    def log_error(self, stage: str, error: str, details: Optional[Dict[str, Any]] = None):
        """Log an error in the pipeline"""
        error_data = {
            "stage": stage,
            "error": error,
            "details": details or {}
        }
        
        self.log_stage(f"❌ ERROR in {stage}", error_data)
    
    def end_trace(self):
        """End the current trace"""
        self._log("\n" + "="*70)
        self._log("INFERENCE TRACE COMPLETED")
        self._log("="*70)
        
        # Also save as JSON for programmatic analysis
        json_file = str(self.trace_file).replace(".log", ".json")
        with open(json_file, "w") as f:
            json.dump(self.current_trace, f, indent=2)
        
        self._log(f"\nTrace saved to: {self.trace_file}")
        self._log(f"JSON trace saved to: {json_file}")
    
    def _log(self, message: str):
        """Write message to trace file"""
        if self.trace_file:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
    
    def _serialize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize data for JSON storage"""
        serialized = {}
        for key, value in data.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                serialized[key] = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            else:
                serialized[key] = value
        return serialized


# Global tracer instance
_tracer = InferenceTracer()

def get_tracer() -> InferenceTracer:
    """Get the global tracer instance"""
    return _tracer

