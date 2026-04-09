"""Shared audio loading and model inference (used by main.py and test_inference.py)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from spafe.features.lfcc import lfcc as spafe_lfcc

logger = logging.getLogger(__name__)

TARGET_SR = 16000
MIN_DURATION_SEC = 1.0

# Default T>1 softens overconfident logits (honest display scores). Override via env or fit_temperature.py.
_DEFAULT_CALIBRATION_TEMPERATURE = 2.0
_DEFAULT_CONFIDENCE_CAP_PERCENT = 95.0


def _calibration_temperature() -> float:
    t = float(os.environ.get("CALIBRATION_TEMPERATURE", str(_DEFAULT_CALIBRATION_TEMPERATURE)))
    return t if t > 0 else _DEFAULT_CALIBRATION_TEMPERATURE


def _confidence_cap_percent() -> float:
    c = float(os.environ.get("CONFIDENCE_CAP_PERCENT", str(_DEFAULT_CONFIDENCE_CAP_PERCENT)))
    return min(max(c, 1.0), 100.0)


def _resample_mono_to_16k(waveform: np.ndarray, orig_sr: int) -> np.ndarray:
    """waveform: 1D float mono."""
    if orig_sr == TARGET_SR:
        return waveform.astype(np.float32, copy=False)
    w = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)
    w = torchaudio.functional.resample(w, orig_sr, TARGET_SR)
    return w.squeeze(0).numpy().astype(np.float32)


def load_audio_from_path(file_path: str | Path) -> np.ndarray:
    """
    Load mono float32 waveform at 16 kHz.
    Prefer soundfile; for .webm or if soundfile fails, use torchaudio.load.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    audio: np.ndarray | None = None
    sr: int | None = None

    use_torchaudio = suffix == ".webm"

    if not use_torchaudio:
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            audio = np.asarray(data, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1).astype(np.float32)
        except Exception as e:
            logger.warning("soundfile read failed, trying torchaudio: %s", e)
            use_torchaudio = True

    if use_torchaudio:
        try:
            wav_tensor, sr_t = torchaudio.load(str(path))
        except Exception as e:
            raise ValueError(f"Could not decode audio file: {e}") from e
        wav = wav_tensor.numpy()
        if wav.ndim == 2 and wav.shape[0] > 1:
            audio = wav.mean(axis=0).astype(np.float32)
        elif wav.ndim == 2:
            audio = wav.squeeze(0).astype(np.float32)
        else:
            audio = wav.astype(np.float32)
        sr = int(sr_t)

    assert audio is not None and sr is not None

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    audio = _resample_mono_to_16k(audio, sr)
    duration = len(audio) / TARGET_SR
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short: need at least {MIN_DURATION_SEC} second(s), got {duration:.2f}s"
        )
    return audio


def extract_lfcc_spafe(audio_waveform: np.ndarray, sample_rate: int = TARGET_SR) -> torch.Tensor:
    """
    LFCC via spafe (matches typical ASVspoof / spafe training pipelines).
    Returns tensor shape (1, 1, num_ceps, time_frames) for FusionModel.
    """
    x = np.asarray(audio_waveform, dtype=np.float32).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)

    features = spafe_lfcc(
        x,
        fs=sample_rate,
        num_ceps=20,
        nfilts=40,
        nfft=512,
        low_freq=0,
        high_freq=sample_rate // 2,
    )
    # spafe: (time_frames, num_ceps) -> model expects (batch, 1, ceps, time)
    lfcc_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32)).T.unsqueeze(0).unsqueeze(0)
    return lfcc_tensor


def run_inference(
    waveform_16k_mono: np.ndarray,
    *,
    model: torch.nn.Module,
    wav2vec_model: torch.nn.Module,
    processor,
    device: torch.device,
) -> dict:
    """
    waveform_16k_mono: 1D float32 numpy, sample rate 16 kHz, length >= 1s.
    Model returns logits; we apply sigmoid for raw prob, sigmoid(logit/T) for display score.
    Returns dict with score, prediction, confidence (no processing_time_ms).
    """
    inputs = processor(
        waveform_16k_mono,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    T = _calibration_temperature()
    cap_pct = _confidence_cap_percent()

    with torch.no_grad():
        w2v_out = wav2vec_model(**inputs)
        wav2vec_feat = w2v_out.last_hidden_state.mean(dim=1)

        lfcc_tensor = extract_lfcc_spafe(waveform_16k_mono).to(device)
        logit_t = model(lfcc_tensor, wav2vec_feat)
        logit = float(logit_t.item())
        prob_raw = float(torch.sigmoid(logit_t).item())
        prob = float(torch.sigmoid(logit_t / T).item())

    prediction = "Fake" if prob > 0.5 else "Real"
    conf_uncapped = float(prob * 100) if prediction == "Fake" else float((1.0 - prob) * 100)
    confidence = min(conf_uncapped, cap_pct)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "score": prob,
        "logit": logit,
        "score_raw": prob_raw,
        "calibration_temperature": T,
        "confidence_cap_percent": cap_pct,
        "probability_note": (
            "Scores use temperature scaling (default T=2) so they are less extreme than raw "
            "model output; confidence is capped. Fit T on validation data with "
            "backend/fit_temperature.py for better calibration."
        ),
    }
