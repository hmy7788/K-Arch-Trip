"""K-Trip API. Team Pipeline reused; serialized model access protects Grad-CAM."""
import asyncio
import base64
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
import os
import threading
import time
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 6 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 16_000_000
lock = threading.Lock()
pipe = None

def decode(data):
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Image must be between 1 byte and 6 MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError("Image is too large (maximum 16 megapixels).")
            im = ImageOps.exif_transpose(source).convert("RGB")
            im.thumbnail((1280, 1280))
            return np.asarray(im)[:, :, ::-1].copy()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Please select a valid JPEG, PNG or WebP image.") from exc

def jpeg(rgb):
    out = BytesIO()
    Image.fromarray(rgb).save(out, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode()

def infer(data, capture=False):
    im = decode(data)
    started = time.perf_counter()
    with lock:
        result = pipe.predict(im)
        result["model_arch"] = pipe.meta["arch"]
        result["resize"] = pipe.meta["resize"]
        if capture:
            explanation = pipe.explain(im, result["label"])
            cam = explanation["cam"]
            zones = [float(part.mean()) for part in np.array_split(cam, 3, axis=0)]
            result.update(overlay=jpeg(explanation["overlay"]), photo=jpeg(im[:, :, ::-1]),
                          focus=["upper", "middle", "lower"][int(np.argmax(zones))],
                          cam_active=bool(float(cam.max()) > 0),
                          uncertain=result["score"] < .70,
                          margin=sorted(result["probs"].values(), reverse=True)[0] - sorted(result["probs"].values(), reverse=True)[1])
    result["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result

@asynccontextmanager
async def lifespan(app):
    global pipe
    pipe = await asyncio.to_thread(Pipeline, Path(os.environ.get("KTRIP_CHECKPOINT", ROOT / "checkpoints")))
    await asyncio.to_thread(pipe.predict, np.zeros((224, 224, 3), dtype=np.uint8))
    yield

app = FastAPI(title="K-Trip", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"ready": pipe is not None, "model": pipe.info() if pipe else None}

@app.post("/api/capture")
async def capture(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "Image exceeds 6 MB.")
    try:
        return await asyncio.to_thread(infer, bytes(data), True)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

@app.websocket("/api/live")
async def live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            try:
                result = await asyncio.to_thread(infer, data)
                await ws.send_json(result)
            except ValueError as exc:
                await ws.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        pass
