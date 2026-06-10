"""체크포인트로부터 FAISS 검색 인덱스와 ONNX 모델을 생성해 저장한다."""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets import AnimeCharacterDataset
from src.inference import AnimeRetriever
from src.models import EfficientNetEmbedder
from src.utils import get_transforms


class _NormalizedWrapper(nn.Module):
    """forward() 출력에 L2 정규화를 포함시켜 ONNX로 내보내는 래퍼."""
    def __init__(self, model: EfficientNetEmbedder):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(x), dim=-1)


def main():
    """데이터셋 전체 임베딩을 추출하고 FAISS 인덱스, 라벨, ONNX 모델을 저장한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--index-out", default="checkpoints/index.faiss")
    parser.add_argument("--labels-out", default="checkpoints/labels.npy")
    parser.add_argument("--onnx-out", default="checkpoints/model.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = EfficientNetEmbedder()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    # ONNX 익스포트 (L2 정규화 포함)
    wrapper = _NormalizedWrapper(model)
    dummy = torch.zeros(1, 3, args.image_size, args.image_size, device=device)
    torch.onnx.export(
        wrapper,
        dummy,
        args.onnx_out,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX saved to {args.onnx_out}")

    # FAISS 인덱스 빌드
    dataset = AnimeCharacterDataset(args.data_root, transform=get_transforms(args.image_size, train=False))
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

    embeddings, label_names = [], []
    with torch.no_grad():
        for images, lbls in loader:
            emb = model.encode(images.to(device)).cpu().numpy()
            embeddings.append(emb)
            label_names.extend([dataset.classes[l] for l in lbls.tolist()])

    embeddings = np.vstack(embeddings)

    retriever = AnimeRetriever(model)
    retriever.build_index(embeddings, label_names)
    retriever.save(args.index_out)
    np.save(args.labels_out, np.array(label_names))
    print(f"Index saved to {args.index_out}  ({len(label_names)} vectors)")
    print(f"Labels saved to {args.labels_out}")


if __name__ == "__main__":
    main()
