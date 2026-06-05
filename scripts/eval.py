"""Build FAISS index and evaluate retrieval metrics."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets import AnimeCharacterDataset
from src.inference import AnimeRetriever
from src.metrics import mean_average_precision, top_k_accuracy
from src.models import EfficientNetEmbedder
from src.utils import get_transforms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = EfficientNetEmbedder()
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    dataset = AnimeCharacterDataset(args.data_root, transform=get_transforms(args.image_size, train=False))
    loader = DataLoader(dataset, batch_size=64, num_workers=4)

    embeddings, labels = [], []
    with torch.no_grad():
        for images, lbls in loader:
            emb = model.encode(images.to(device)).cpu().numpy()
            embeddings.append(emb)
            labels.extend(lbls.tolist())

    embeddings = np.vstack(embeddings)
    labels_arr = np.array(labels)

    retriever = AnimeRetriever(model)
    retriever.build_index(embeddings.copy(), [dataset.classes[l] for l in labels])

    retrieved = []
    for emb in embeddings:
        scores, idxs = retriever.index.search(emb[None], args.top_k)
        retrieved.append(labels_arr[idxs[0]])

    top1 = top_k_accuracy(retrieved, labels_arr, k=1)
    top5 = top_k_accuracy(retrieved, labels_arr, k=5)
    mAP = mean_average_precision(retrieved, labels_arr)
    print(f"Top-1: {top1:.4f}  Top-5: {top5:.4f}  mAP: {mAP:.4f}")


if __name__ == "__main__":
    main()
