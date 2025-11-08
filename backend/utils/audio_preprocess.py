"""Exact reproduction of Phase 1B training preprocessing."""

import torch
import torchaudio
import numpy as np
import noisereduce as nr
import torch.nn.functional as F


SAMPLE_RATE = 22050
TARGET_DURATION = 4.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DESIRED_TIME_FRAMES = 128


mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
)
amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")


TRAINING_PARAMS = {
    "SAMPLE_RATE": SAMPLE_RATE,
    "N_FFT": N_FFT,
    "HOP_LENGTH": HOP_LENGTH,
    "N_MELS": N_MELS,
    "TARGET_DURATION": TARGET_DURATION,
    "TARGET_SAMPLES": TARGET_SAMPLES,
    "DESIRED_TIME_FRAMES": DESIRED_TIME_FRAMES,
    "NOISE_REDUCTION": True,
    "RMS_NORMALIZATION": True,
    "PAD_TRIM": True,
    "SCALE": "dB",
    "NORMALIZATION": "zscore",
}


def get_preprocessing_params() -> dict:
    return TRAINING_PARAMS.copy()


def safe_audio(y: np.ndarray) -> np.ndarray:
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    rms = np.sqrt(np.mean(y**2))
    return y / (rms + 1e-8) if rms > 0 else y


def pad_or_trim(y: np.ndarray) -> np.ndarray:
    if len(y) > TARGET_SAMPLES:
        return y[:TARGET_SAMPLES]
    return np.pad(y, (0, TARGET_SAMPLES - len(y)))


def clean_and_standardize_waveform(y: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    y = safe_audio(y)

    try:
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)
    except Exception:
        pass

    y = safe_audio(y)
    return pad_or_trim(y)


def preprocess_audio_tensor(
    y: np.ndarray,
    sr: int,
    device: str | torch.device | None = None,
    tracer=None,
    is_live: bool = False,
):
    if y.ndim > 1:
        y = y.mean(axis=-1)

    y = clean_and_standardize_waveform(y, sr)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

    mel = mel_transform(waveform)
    mel_db = amp_to_db(mel)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    mel_db_4d = mel_db.unsqueeze(0)
    mel_resized = F.interpolate(
        mel_db_4d, size=(N_MELS, DESIRED_TIME_FRAMES), mode="bilinear", align_corners=False
    )
    mel_final = mel_resized.squeeze(0)

    return mel_final


def preprocess_audio(file_path: str, device: str | torch.device | None = None):
    waveform, sr = torchaudio.load(file_path)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)
    waveform_np = waveform.cpu().numpy()
    return preprocess_audio_tensor(waveform_np, sr, device=device)


def get_audio_duration(file_path: str) -> float:
    info = torchaudio.info(file_path)
    return info.num_frames / info.sample_rate


def verify_alignment(sample_wav: str, sample_npy: str) -> None:
    import soundfile as sf

    wav, sr = sf.read(sample_wav, dtype="float32")
    mel_tensor = preprocess_audio_tensor(wav, sr, device="cpu")
    mel_backend = mel_tensor.squeeze().cpu().numpy()

    mel_train = np.load(sample_npy)
    diff = np.mean(np.abs(mel_backend - mel_train))
    print(f"Δ mean absolute diff: {diff:.6f}")
