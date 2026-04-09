"""Load models and run the full inference pipeline on dummy 3s silence (smoke test)."""

import numpy as np

import main
from inference import run_inference


def main_smoke() -> None:
    wave = np.zeros(3 * 16000, dtype=np.float32)
    out = run_inference(wave, 16000, model=main.model, device=main.device)
    print("prediction:", out["prediction"])
    print("score (P spoof):", out["score"])
    print("confidence (% for predicted class):", out["confidence"])


if __name__ == "__main__":
    main_smoke()
