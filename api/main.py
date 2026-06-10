"""FastAPI 서버 — 이미지 업로드 → ONNX 추론 → FAISS 검색 → 캐릭터 목록 반환."""
import io
import json
import os
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

DATA_ROOT = Path(os.getenv("DATA_ROOT", "data/processed"))
CHAR_NAMES_PATH = Path(os.getenv("CHAR_NAMES_PATH", "data/character_names.json"))
ONNX_PATH = Path(os.getenv("ONNX_PATH", "checkpoints/model.onnx"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "checkpoints/index.faiss"))
LABELS_PATH = Path(os.getenv("LABELS_PATH", "checkpoints/labels.npy"))
TOP_K = int(os.getenv("TOP_K", "5"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "224"))

_session: ort.InferenceSession | None = None
_index: faiss.IndexFlatIP | None = None
_labels: list[str] = []
_input_name: str = "image"
_name_map: dict[str, str] = {}


def _display(label: str) -> str:
    return _name_map.get(unicodedata.normalize("NFC", label), label)

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

    if CHAR_NAMES_PATH.exists():
        _name_map.update(json.loads(CHAR_NAMES_PATH.read_text(encoding="utf-8")))
        print(f"[OK] character names loaded  ({len(_name_map)} entries)")

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

if DATA_ROOT.exists():
    app.mount("/images", StaticFiles(directory=str(DATA_ROOT)), name="images")

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
    unique = sorted({_display(l) for l in _labels})
    return {"model_loaded": True, "characters": unique, "total": len(unique)}


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@app.get("/characters/{character}/images")
def character_images(character: str):
    # 한글 표시명 → 원래 폴더명 역매핑
    reverse = {v: k for k, v in _name_map.items()}
    folder = reverse.get(character, character)
    char_dir = DATA_ROOT / folder
    if not char_dir.exists():
        raise HTTPException(404, f"Character '{character}' not found")
    from urllib.parse import quote
    imgs = sorted(
        f"/api/images/{quote(folder)}/{quote(f.name)}"
        for f in char_dir.iterdir()
        if f.suffix.lower() in _IMAGE_EXTS
    )
    return {"character": character, "images": imgs[:20]}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _session is None:
        return PredictResponse(model_loaded=False, results=[])

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = _preprocess(image)

    embedding = _session.run(None, {_input_name: tensor})[0].astype(np.float32)
    faiss.normalize_L2(embedding)
    # TOP_K * 10 후보 검색 → 캐릭터별 최고 점수로 집계 → 상위 TOP_K 반환
    k_search = min(TOP_K * 10, _index.ntotal)
    scores, indices = _index.search(embedding, k_search)

    best: dict[str, float] = {}
    for rank, idx in enumerate(indices[0]):
        if idx >= len(_labels):
            continue
        char = _display(_labels[idx])
        score = float(scores[0][rank])
        if char not in best or score > best[char]:
            best[char] = score

    results = [
        Match(character=char, score=round(score, 4))
        for char, score in sorted(best.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    ]
    return PredictResponse(model_loaded=True, results=results)
