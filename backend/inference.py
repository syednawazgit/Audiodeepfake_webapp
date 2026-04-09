"""Shared audio loading and model inference (used by main.py and test_inference.py)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from spafe.features.lfcc import lfcc as spafe_lfcc

logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 1.0

_FFMPEG_TIMEOUT_SEC = 120


def _ffmpeg_executable() -> str | None:
    """System ffmpeg on PATH, else binary shipped with imageio-ffmpeg (no TorchCodec needed)."""
    w = shutil.which("ffmpeg")
    if w:
        return w
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _load_via_ffmpeg(path: Path, ffmpeg_exe: str) -> tuple[np.ndarray, int]:
    """Decode almost any format (WebM/Opus, etc.) to mono float32 WAV via ffmpeg; preserves input sample rate."""
    fd, out_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        r = subprocess.run(
            [
                ffmpeg_exe,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-f",
                "wav",
                "-acodec",
                "pcm_f32le",
                out_wav,
            ],
            capture_output=True,
            text=True,
            timeout=_FFMPEG_TIMEOUT_SEC,
            check=False,
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout or "").strip() or f"exit {r.returncode}"
            raise RuntimeError(err)
        data, sr = sf.read(out_wav, dtype="float32", always_2d=False)
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1).astype(np.float32)
        return audio, int(sr)
    finally:
        try:
            Path(out_wav).unlink(missing_ok=True)
        except OSError:
            pass


def _load_via_torchaudio(path: Path) -> tuple[np.ndarray, int]:
    wav_tensor, sr_t = torchaudio.load(str(path))
    wav = wav_tensor.numpy()
    if wav.ndim == 2 and wav.shape[0] > 1:
        audio = wav.mean(axis=0).astype(np.float32)
    elif wav.ndim == 2:
        audio = wav.squeeze(0).astype(np.float32)
    else:
        audio = wav.astype(np.float32)
    return audio, int(sr_t)


def load_audio_from_path(file_path: str | Path) -> tuple[np.ndarray, int]:
    """
    Load mono float32 waveform and its sample rate (native; no resampling).
    Same idea as standalone `sf.read` + mono: LFCC must use this `sr` in `lfcc(..., fs=sr)`.

    Order: soundfile (non-WebM) → ffmpeg (WebM / fallback; avoids TorchCodec) → torchaudio.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    audio: np.ndarray | None = None
    sr: int | None = None

    if suffix != ".webm":
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            audio = np.asarray(data, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1).astype(np.float32)
        except Exception as e:
            logger.warning("soundfile read failed, will try ffmpeg/torchaudio: %s", e)

    ffmpeg_error: BaseException | None = None
    if audio is None:
        ff = _ffmpeg_executable()
        if ff:
            try:
                audio, sr = _load_via_ffmpeg(path, ff)
            except Exception as e:
                ffmpeg_error = e
                logger.warning("ffmpeg decode failed: %s", e)
        if audio is None and suffix == ".webm":
            raise ValueError(
                "Could not decode WebM (browser recording). "
                "Run: pip install imageio-ffmpeg (bundled ffmpeg), or install FFmpeg and add it to PATH."
            ) from ffmpeg_error
        if audio is None:
            try:
                audio, sr = _load_via_torchaudio(path)
            except Exception as e:
                raise ValueError(
                    "Could not decode audio file: "
                    f"{e}. Try: pip install imageio-ffmpeg, or put ffmpeg on PATH."
                ) from e

    assert audio is not None and sr is not None

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)

    sr_i = int(sr)
    duration = len(audio) / float(sr_i)
    if duration < MIN_DURATION_SEC:
        raise ValueError(
            f"Audio too short: need at least {MIN_DURATION_SEC} second(s), got {duration:.2f}s"
        )
    return audio, sr_i


def features_from_waveform(audio_waveform: np.ndarray, sample_rate: int) -> torch.Tensor:
    """
    Same LFCC call chain as standalone single-file eval: mono waveform → spafe → (n_ceps, T) → batch.
    """
    x = np.asarray(audio_waveform, dtype=np.float32).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)

    # Match eval script: torch tensor in/out so behavior matches `wav.squeeze().numpy()` path.
    w = torch.from_numpy(x).unsqueeze(0)
    wav = w.squeeze().numpy()

    features = spafe_lfcc(
        wav,
        fs=sample_rate,
        num_ceps=20,
        nfilts=40,
        nfft=512,
        low_freq=0,
        high_freq=sample_rate // 2,
    )
    # (time_frames, n_ceps) -> (1, n_ceps, time_frames); torch.tensor like training/eval scripts
    feat = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32).T.unsqueeze(0)
    return feat


def run_inference(
    waveform_mono: np.ndarray,
    sample_rate: int,
    *,
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    waveform_mono: 1D float32 numpy; sample_rate must match how the file was decoded (native rate).
    Returns model P(spoof) as `score` and label confidence (%) for the predicted class, uncapped.
    """
    feat = features_from_waveform(waveform_mono, sample_rate).to(device)

    with torch.no_grad():
        prob_t = model(feat).squeeze()
        prob = float(prob_t.item())

    prediction = "Fake" if prob > 0.5 else "Real"
    confidence = float(prob * 100.0) if prediction == "Fake" else float((1.0 - prob) * 100.0)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "score": prob,
    }
