"""Load models and run the full inference pipeline on dummy 3s silence (smoke test)."""

import numpy as np

import main
from inference import run_inference


def main_smoke() -> None:
    wave = np.zeros(3 * 16000, dtype=np.float32)
    out = run_inference(
        wave,
        model=main.model,
        wav2vec_model=main.wav2vec_model,
        processor=main.processor,
        device=main.device,
    )
    print("prediction:", out["prediction"])
    print("score (calibrated P fake):", out["score"])
    print("score_raw (uncalibrated):", out.get("score_raw"))
    print("logit:", out.get("logit"))
    print("confidence:", out["confidence"])


if __name__ == "__main__":
    main_smoke()
