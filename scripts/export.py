"""체크포인트로부터 FAISS 검색 인덱스를 생성해 저장한다."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import AnimeCharacterDataset
from src.inference import AnimeRetriever
from src.models import EfficientNetEmbedder
from src.utils import get_transforms


def main():
    """데이터셋 전체 임베딩을 추출하고 라벨 이름과 함께 인덱싱한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--index-out", default="checkpoints/index.faiss")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = EfficientNetEmbedder()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

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
    print(f"Index saved to {args.index_out}  ({len(label_names)} vectors)")


if __name__ == "__main__":
    main()
