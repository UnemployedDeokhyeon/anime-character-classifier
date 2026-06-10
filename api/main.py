"""FastAPI 서버 — 이미지 업로드 → ONNX 추론 → FAISS 검색 → 캐릭터 목록 반환."""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

ONNX_PATH = Path(os.getenv("ONNX_PATH", "checkpoints/model.onnx"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "checkpoints/index.faiss"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "checkpoints/labels.npy"))
TOP_K = int(os.getenv("TOP_K", "5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

_session: ort.InferenceSession | None = None
_index: faiss.IndexFlatIP | None = None
_labels: list[str] = []
_input_name: str = "image"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    return arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session, _index, _labels, _input_name

    missing = [str(p) for p in [ONNX_PATH, INDEX_PATH, LABELS_PATH] if not p.exists()]
    if missing:
        print(f"[WARN] missing: {missing}  →  run scripts/export.py first")
    else:
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        _index = faiss.read_index(str(INDEX_PATH))
        _labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
        print(f"[OK] ONNX loaded  index={_index.ntotal}  labels={len(_labels)}")

    yield


app = FastAPI(title="Anime Character Finder", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Match(BaseModel):
    character: str
    score: float


class PredictResponse(BaseModel):
    model_loaded: bool
    results: list[Match]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _session is not None}


@app.get("/characters")
def characters():
    if _session is None:
        return {"model_loaded": False, "characters": []}
    unique = sorted(set(_labels))
    return {"model_loaded": True, "characters": unique, "total": len(unique)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _session is None:
        return PredictResponse(model_loaded=False, results=[])

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = _preprocess(image)

    embedding = _session.run(None, {_input_name: tensor})[0].astype(np.float32)
    faiss.normalize_L2(embedding)
    scores, indices = _index.search(embedding, TOP_K)

    results = [
        Match(character=_labels[idx], score=round(float(scores[0][rank]), 4))
        for rank, idx in enumerate(indices[0])
        if idx < len(_labels)
    ]
    return PredictResponse(model_loaded=True, results=results)
