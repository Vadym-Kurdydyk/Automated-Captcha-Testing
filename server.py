# Local web API + pages for trying the model and for end-to-end captcha testing.
#
#   python server.py                                   -> http://127.0.0.1:8000
#   CAPTCHA_MODEL=checkpoints/x/best.pth CAPTCHA_DEVICE=cpu python server.py
#
#   GET  /                      upload page (drag & drop images, see predictions)
#   POST /api/predict           multipart "files" -> [{filename, text, confidence, answer, low_confidence}]
#   GET  /challenge             a captcha form served from our own generator (for Playwright tests)
#   GET  /api/challenge         {id, image (data URL), difficulty}
#   POST /api/verify            {id, answer} -> {correct, expected}

import base64
import io
import itertools
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from captchagen import CaptchaGenerator, GeneratorConfig
from predict import CaptchaReader, default_checkpoint

MAX_FILES = 16
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000            # refuse decompression bombs / huge photos
LOW_CONFIDENCE = 0.5
CHALLENGE_TTL = 300               # seconds a challenge stays answerable
WEB = Path(__file__).parent / "web"

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class State:
    reader: CaptchaReader
    checkpoint: str
    generator: CaptchaGenerator
    counter = itertools.count()
    challenges: dict[str, tuple[str, float, int]] = {}   # id -> (text, created, difficulty)
    lock = threading.Lock()                               # one inference at a time


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.checkpoint = os.environ.get("CAPTCHA_MODEL") or default_checkpoint()
    state.reader = CaptchaReader(state.checkpoint, os.environ.get("CAPTCHA_DEVICE"))
    # Challenges use the same generator config the model was trained for, with an unguessable seed
    cfg = {k: v for k, v in state.reader.generator_config.items() if k != "vocab"}
    state.generator = CaptchaGenerator(GeneratorConfig(**cfg), seed=secrets.randbits(31))
    print(f"Model {state.checkpoint} on {state.reader.device}")
    yield


app = FastAPI(title="Captcha model", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/challenge")
def challenge_page():
    return FileResponse(WEB / "challenge.html")


@app.get("/api/info")
def info():
    return {"checkpoint": state.checkpoint, "device": str(state.reader.device), "vocab": state.reader.vocab.chars,
            "math": state.reader.is_math}


@app.post("/api/predict")
def predict(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"At most {MAX_FILES} files per request")
    images, names = [], []
    for f in files:
        data = f.file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename}: larger than {MAX_FILE_BYTES // 1024} KB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError):
            raise HTTPException(400, f"{f.filename}: not a readable image (or too many pixels)")
        images.append(img.convert("RGB"))
        names.append(f.filename)
    with state.lock:
        preds = state.reader.read(images)
    return [{"filename": n, "text": p.text, "confidence": round(p.confidence, 4), "answer": p.answer,
             "low_confidence": p.confidence < LOW_CONFIDENCE, "size": [img.width, img.height]}
            for n, p, img in zip(names, preds, images)]


@app.get("/api/challenge")
def new_challenge(difficulty: int = Query(3, ge=1, le=5)):
    now = time.time()
    with state.lock:
        for cid in [c for c, (_, t, _) in state.challenges.items() if now - t > CHALLENGE_TTL]:
            del state.challenges[cid]
        sample = state.generator.generate(next(state.counter), difficulty)
    cid = secrets.token_urlsafe(12)
    expected = str(sample.answer) if sample.answer is not None else sample.text
    state.challenges[cid] = (expected, now, difficulty)
    buf = io.BytesIO()
    sample.image.save(buf, "PNG")
    return {"id": cid, "difficulty": difficulty,
            "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}


class Answer(BaseModel):
    id: str
    answer: str


@app.post("/api/verify")
def verify(body: Answer):
    entry = state.challenges.pop(body.id, None)   # one attempt per challenge
    if entry is None or time.time() - entry[1] > CHALLENGE_TTL:
        raise HTTPException(404, "Unknown or expired challenge")
    expected, _, difficulty = entry
    return {"correct": body.answer.strip() == expected, "expected": expected, "difficulty": difficulty}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
