from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

from src.models import EfficientNetEmbedder
from src.utils import get_transforms


class AnimeRetriever:
    """FAISS-backed retrieval: query image → top-k matching characters."""

    def __init__(self, model: EfficientNetEmbedder, embedding_dim: int = 512):
        self.model = model
        self.model.eval()
        self.transform = get_transforms(train=False)
        self.index = faiss.IndexFlatIP(embedding_dim)  # inner product = cosine on L2-normed vecs
        self.labels: list[str] = []
        
    # 인덱스 구축
    def build_index(self, embeddings: np.ndarray, labels: list[str]) -> None:
        self.index.reset()
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.labels = labels
        
    # 쿼리 검색
    def search(self, image_path: str | Path, top_k: int = 10) -> list[tuple[str, float]]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        embedding = self.model.encode(tensor).cpu().numpy()
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, top_k)
        return [(self.labels[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]

    def save(self, path: str | Path) -> None:
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        self.index = faiss.read_index(str(path))
