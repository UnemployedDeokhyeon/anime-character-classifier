from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

from src.models import EfficientNetEmbedder
from src.utils import get_transforms


class AnimeRetriever:
    """FAISS 인덱스로 쿼리 이미지와 가까운 캐릭터를 검색한다."""

    def __init__(self, model: EfficientNetEmbedder, embedding_dim: int = 512):
        self.model = model
        self.model.eval()
        self.transform = get_transforms(train=False)
        # L2 정규화된 벡터에서는 내적이 코사인 유사도와 같다.
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.labels: list[str] = []
        
    def build_index(self, embeddings: np.ndarray, labels: list[str]) -> None:
        """임베딩 배열을 정규화한 뒤 FAISS 인덱스에 등록한다."""
        self.index.reset()
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.labels = labels
        
    def search(self, image_path: str | Path, top_k: int = 10) -> list[tuple[str, float]]:
        """이미지 경로 하나를 받아 top-k 캐릭터 라벨과 점수를 반환한다."""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        embedding = self.model.encode(tensor).cpu().numpy()
        faiss.normalize_L2(embedding)
        scores, indices = self.index.search(embedding, top_k)
        return [(self.labels[i], float(scores[0][rank])) for rank, i in enumerate(indices[0])]

    def save(self, path: str | Path) -> None:
        """현재 FAISS 인덱스를 파일로 저장한다."""
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        """저장된 FAISS 인덱스를 로드한다."""
        self.index = faiss.read_index(str(path))
