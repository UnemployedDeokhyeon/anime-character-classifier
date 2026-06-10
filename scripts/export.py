"""체크포인트로부터 FAISS 검색 인덱스와 ONNX 모델을 생성해 저장한다.

두 가지 체크포인트 포맷 지원:
  기본값  — EfficientNetEmbedder 형식 {"model_state": ...}  (scripts/train.py 산출물)
  --backbone 지정 시 — raw timm 분류 모델 .pth (scripts/compare_models.py 산출물)
"""
import argparse

import numpy as np
import timm
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

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.model(x), dim=-1)


def load_model(args) -> tuple[nn.Module, int]:
    """체크포인트를 로드하고 (추론용 모델, embedding_dim) 반환.

    --backbone 지정 시 raw timm 분류 모델로 로드하고 classifier를 제거한다.
    미지정 시 EfficientNetEmbedder {"model_state": ...} 포맷으로 로드한다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.backbone:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
        num_classes = state_dict["classifier.weight"].shape[0]
        model = timm.create_model(args.backbone, pretrained=False, num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.reset_classifier(0)  # classifier 제거 → backbone feature 출력
        embedding_dim = model.num_features
    else:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model = EfficientNetEmbedder()
        model.load_state_dict(ckpt["model_state"])
        embedding_dim = 512  # EfficientNetEmbedder projector 출력 dim

    model.to(device).eval()
    return model, embedding_dim, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default=None,
                        help="timm 모델명 지정 시 raw timm 분류 모델로 로드 (예: efficientnet_b4)")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--index-out", default="checkpoints/index.faiss")
    parser.add_argument("--labels-out", default="checkpoints/labels.npy")
    parser.add_argument("--onnx-out", default="checkpoints/model.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    model, embedding_dim, device = load_model(args)
    print(f"Model loaded  embedding_dim={embedding_dim}")

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
    dataset = AnimeCharacterDataset(
        args.data_root, transform=get_transforms(args.image_size, train=False)
    )
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

    embeddings, label_names = [], []
    with torch.no_grad():
        for images, lbls in loader:
            if args.backbone:
                emb = F.normalize(model(images.to(device)), dim=-1).cpu().numpy()
            else:
                emb = model.encode(images.to(device)).cpu().numpy()
            embeddings.append(emb)
            label_names.extend([dataset.classes[l] for l in lbls.tolist()])

    embeddings = np.vstack(embeddings)

    retriever = AnimeRetriever(model, embedding_dim)
    retriever.build_index(embeddings, label_names)
    retriever.save(args.index_out)
    np.save(args.labels_out, np.array(label_names))
    print(f"Index saved to {args.index_out}  ({len(label_names)} vectors)")
    print(f"Labels saved to {args.labels_out}")


if __name__ == "__main__":
    main()
