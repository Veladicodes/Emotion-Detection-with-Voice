import os, torch, soundfile as sf, numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from backend.utils.audio_preprocess import preprocess_audio_tensor
from backend.utils.model_loader import get_model

# ==========================================================
# CONFIG
# ==========================================================
RAW_DIR = r"D:\Emotion Voice\Data\raw"
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
MODEL_PATH = r"D:\Emotion Voice\phase3_outputs\final_export\best_efficientnet_v2_m+vgg16+densenet121.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================================
# LOAD MODEL (already softmaxed ensemble)
# ==========================================================
print(f"🟢 Using device: {DEVICE}")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval().to(DEVICE)

# ==========================================================
# EVALUATION LOOP
# ==========================================================
y_true, y_pred = [], []
print("\n🎧 Evaluating on full dataset...\n")

for emotion in EMOTIONS:
    emotion_dir = os.path.join(RAW_DIR, emotion)
    if not os.path.isdir(emotion_dir):
        print(f"⚠️ Missing folder: {emotion}")
        continue

    files = [f for f in os.listdir(emotion_dir) if f.lower().endswith(".wav")]
    for f in tqdm(files, desc=f"Testing {emotion}", ncols=100):
        try:
            path = os.path.join(emotion_dir, f)
            wav, sr = sf.read(path, dtype="float32")
            mel_tensor = preprocess_audio_tensor(wav, sr, device=DEVICE)

            if mel_tensor.dim() == 3:
                mel_tensor = mel_tensor.unsqueeze(0)  # (1,1,128,128)
            mel_tensor = mel_tensor.to(DEVICE)

            with torch.no_grad():
                out = model(mel_tensor)          # Already softmaxed inside
                probs = out[0].cpu()
                probs = probs / probs.sum()      # Normalize (no double softmax)
                pred_idx = torch.argmax(probs).item()

            y_true.append(emotion)
            y_pred.append(EMOTIONS[pred_idx])

        except Exception as e:
            print(f"⚠️ Error processing {f}: {e}")

# ==========================================================
# METRICS
# ==========================================================
acc = accuracy_score(y_true, y_pred)
print("\n✅ Overall Accuracy:", round(acc * 100, 2), "%\n")
print("📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=EMOTIONS, digits=3))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred, labels=EMOTIONS))
