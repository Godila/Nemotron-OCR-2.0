"""
Nemotron OCR v2 — FastAPI server
Эндпоинты /predict и /healthCheck совместимы с CAILA.
"""
import base64
import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ocr_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_model
    logger.info("Loading Nemotron OCR v2...")
    try:
        import torch
        logger.info("CUDA available: %s, device count: %d",
                    torch.cuda.is_available(), torch.cuda.device_count())
        from nemotron_ocr.inference.pipeline import NemotronOCR
        ocr_model = NemotronOCR()
        logger.info("Model ready.")
    except Exception as e:
        logger.exception("FATAL: model failed to load: %s", e)
        raise
    yield
    ocr_model = None


app = FastAPI(title="Nemotron OCR v2", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Схемы ────────────────────────────────────────────────────────────────────

class BBoxVertex(BaseModel):
    x: int
    y: int


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: List[BBoxVertex]


class OCRRequest(BaseModel):
    image_b64: str
    mime: str = "image/jpeg"
    merge_level: str = "paragraph"


class OCRResponse(BaseModel):
    full_text: str
    lines: List[OCRLine]
    avg_confidence: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/healthCheck")
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ocr_model is not None}


@app.post("/predict", response_model=OCRResponse)
@app.post("/ocr", response_model=OCRResponse)
async def predict(req: OCRRequest):
    if ocr_model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    suffix = ".jpg" if "jpeg" in req.mime else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        merge = req.merge_level if req.merge_level in ("layout", "word", "sentence", "paragraph") else "paragraph"
        predictions = ocr_model(tmp_path, merge_level=merge)
        logger.info("merge_level=%s", merge)
    except RuntimeError as e:
        os.unlink(tmp_path)
        if "doesn't have storage" in str(e) or "Cannot access data pointer" in str(e):
            logger.warning("OCR: no text regions detected, returning empty result")
            return OCRResponse(full_text="", lines=[], avg_confidence=0.0)
        logger.exception("OCR error: %s", e)
        raise HTTPException(500, str(e))
    except Exception as e:
        os.unlink(tmp_path)
        logger.exception("OCR error: %s", e)
        raise HTTPException(500, str(e))
    else:
        os.unlink(tmp_path)

    img = Image.open(io.BytesIO(img_bytes))
    img_w, img_h = img.size

    lines: List[OCRLine] = []
    texts = []

    for pred in predictions:
        text = str(pred.get("text", "") or "")
        conf = float(pred.get("confidence", 1.0))
        x1 = int(pred.get("left",  0.0) * img_w)
        x2 = int(pred.get("right", 1.0) * img_w)
        y1 = int(pred.get("lower", 0.0) * img_h)
        y2 = int(pred.get("upper", 1.0) * img_h)
        bbox = [
            BBoxVertex(x=x1, y=y1), BBoxVertex(x=x2, y=y1),
            BBoxVertex(x=x2, y=y2), BBoxVertex(x=x1, y=y2),
        ]
        if text.strip():
            lines.append(OCRLine(text=text, confidence=conf, bbox=bbox))
            texts.append(text)

    full_text = "\n".join(texts)
    avg_conf = sum(l.confidence for l in lines) / len(lines) if lines else 0.0
    logger.info("OCR done: %d lines, avg_conf=%.3f", len(lines), avg_conf)
    return OCRResponse(full_text=full_text, lines=lines, avg_confidence=avg_conf)


if __name__ == "__main__":
    port = int(os.environ.get("SERVICE_PORT", "8001"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
