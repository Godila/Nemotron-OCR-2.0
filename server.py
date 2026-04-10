"""
Nemotron OCR v2 — FastAPI wrapper
Интерфейс совместим с YandexOCRService нашего backend.

Запуск:
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
import base64
import io
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный экземпляр модели
ocr_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ocr_model
    logger.info("Loading Nemotron OCR v2 multilingual...")
    from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    ocr_model = NemotronOCRV2(lang="multi")
    logger.info("Model ready.")
    yield
    ocr_model = None


app = FastAPI(title="Nemotron OCR v2 Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Схемы (совместимы с нашим backend) ───────────────────────────────────────

class BBoxVertex(BaseModel):
    x: int
    y: int


class OCRLine(BaseModel):
    text: str
    confidence: float
    bbox: list[BBoxVertex]


class OCRResult(BaseModel):
    full_text: str
    lines: list[OCRLine]
    avg_confidence: float


class OCRRequest(BaseModel):
    image_b64: str   # base64-encoded image bytes
    mime: str        # image/jpeg | image/png | image/webp
    merge_level: str = "paragraph"  # "layout" | "word" | "sentence" | "paragraph"


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": ocr_model is not None}


@app.post("/ocr", response_model=OCRResult)
async def run_ocr(req: OCRRequest):
    if ocr_model is None:
        raise HTTPException(503, "Model not loaded")

    try:
        img_bytes = base64.b64decode(req.image_b64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image")

    import tempfile, os
    from PIL import Image

    # Сохраняем во временный файл (Nemotron принимает путь к файлу)
    suffix = ".jpg" if "jpeg" in req.mime else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(img_bytes)
        tmp_path = tmp.name

    try:
        # Запускаем OCR в sentence режиме — построчно, хорошее качество
        merge = req.merge_level if req.merge_level in ("layout", "word", "sentence", "paragraph") else "paragraph"
        predictions = ocr_model(tmp_path, merge_level=merge)
        logger.info("merge_level=%s", merge)
    except Exception as e:
        logger.exception("OCR error: %s", e)
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

    # Получаем размеры изображения для нормализации bbox
    from PIL import Image as PILImage
    img = PILImage.open(io.BytesIO(img_bytes))
    img_w, img_h = img.size

    lines: list[OCRLine] = []
    texts: list[str] = []

    for pred in predictions:
        # predictions — list of dicts
        text = str(pred.get("text", "") or "")
        conf = float(pred.get("confidence", 1.0))

        # Координаты нормализованные (0-1):
        # left/right — X; lower = y_min (top), upper = y_max (bottom)
        x1 = int(pred.get("left",  0.0) * img_w)
        x2 = int(pred.get("right", 1.0) * img_w)
        y1 = int(pred.get("lower", 0.0) * img_h)  # top
        y2 = int(pred.get("upper", 1.0) * img_h)  # bottom

        bbox = [
            BBoxVertex(x=x1, y=y1),
            BBoxVertex(x=x2, y=y1),
            BBoxVertex(x=x2, y=y2),
            BBoxVertex(x=x1, y=y2),
        ]

        if text.strip():
            lines.append(OCRLine(text=text, confidence=conf, bbox=bbox))
            texts.append(text)

    full_text = "\n".join(texts)
    avg_conf = sum(l.confidence for l in lines) / len(lines) if lines else 0.0

    logger.info("OCR done: %d lines, avg_conf=%.3f", len(lines), avg_conf)
    return OCRResult(full_text=full_text, lines=lines, avg_confidence=avg_conf)


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8001, log_level="info")
