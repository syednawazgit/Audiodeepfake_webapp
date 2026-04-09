import base64
import logging
import os
import tempfile
import time
from pathlib import Path

import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference import load_audio_from_path, run_inference
from model import AudioBinaryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(BACKEND_DIR / ".env")
_model_env = os.environ.get("MODEL_PATH", "../trained_models/best_lfcc_model.pth")
MODEL_PATH = Path(_model_env)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = (BACKEND_DIR / MODEL_PATH).resolve()

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
BASE64_ALLOWED = {".wav", ".mp3", ".flac", ".ogg", ".webm"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioBinaryClassifier()
try:
    state = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
except TypeError:
    state = torch.load(str(MODEL_PATH), map_location=device)
model.load_state_dict(state)
model.to(device)
model.eval()

app = FastAPI(title="Audio Deepfake Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Base64AudioBody(BaseModel):
    audio: str = Field(..., description="Base64-encoded audio bytes")
    format: str = Field(..., description="File format extension, e.g. wav or webm")


def _suffix_from_upload(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _predict_from_file_path(path: Path) -> dict:
    t0 = time.perf_counter()
    try:
        waveform, sr = load_audio_from_path(path)
        out = run_inference(waveform, sr, model=model, device=device)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Model inference failed") from e
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    out["processing_time_ms"] = elapsed_ms
    return out


@app.get("/health")
def health():
    cuda = torch.cuda.is_available()
    return {
        "status": "ok",
        "device": str(device),
        "cuda_available": cuda,
        "model_loaded": True,
        "model_path": MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = _suffix_from_upload(file.filename)
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return _predict_from_file_path(Path(tmp))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Predict upload failed")
        raise HTTPException(status_code=500, detail="Failed to process upload") from e
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass


@app.post("/predict-base64")
def predict_base64(body: Base64AudioBody):
    fmt = body.format.strip().lower().lstrip(".")
    suffix = f".{fmt}" if fmt else ""
    if suffix not in BASE64_ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(BASE64_ALLOWED))}",
        )
    try:
        raw = base64.b64decode(body.audio, validate=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid base64 audio") from e
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        return _predict_from_file_path(Path(tmp))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Predict base64 failed")
        raise HTTPException(status_code=500, detail="Failed to process audio") from e
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
