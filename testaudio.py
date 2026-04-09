"""
CLI: test a single audio file with the same pipeline as the API (env MODEL_PATH).
Run from repo folder: python testaudio.py <path-to-audio>
"""

import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from inference import load_audio_from_path, run_inference  # noqa: E402
from model import AudioBinaryClassifier  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_model_env = os.environ.get("MODEL_PATH", "../trained_models/best_lfcc_model.pth")
MODEL_PATH = Path(_model_env)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BACKEND_DIR / MODEL_PATH).resolve()

print("Loading model...")
model = AudioBinaryClassifier().to(device)
if not MODEL_PATH.is_file():
    print(f"Error: model file not found: {MODEL_PATH}")
    sys.exit(1)
try:
    state = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
except TypeError:
    state = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(state)
model.eval()
print("Model loaded.\n")


def test_audio(audio_path: str) -> None:
    print(f"Testing: {audio_path}")
    print("-" * 50)
    path = Path(audio_path)
    if not path.is_file():
        print(f"Error: file not found: {path}")
        return
    try:
        waveform, sr = load_audio_from_path(path)
    except ValueError as e:
        print(f"Error: {e}")
        return
    out = run_inference(waveform, sr, model=model, device=device)
    prob = out["score"]
    print("-" * 50)
    print("\nRESULTS:")
    print("=" * 50)
    confidence = max(prob, 1.0 - prob) * 100.0
    if prob > 0.5:
        print("FAKE (SPOOF) DETECTED")
        print(f"   Spoof confidence: {prob * 100:.2f}%")
    else:
        print("GENUINE (BONAFIDE)")
        print(f"   Bonafide confidence: {(1.0 - prob) * 100:.2f}%")
    print("=" * 50)
    print(f"Raw score: {prob:.4f}")
    print(f"Overall confidence: {confidence:.2f}%\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testaudio.py <audio_file_path>")
        sys.exit(1)
    test_audio(sys.argv[1])
