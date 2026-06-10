"""쿼리 이미지 N장을 FAISS 검색해 top-k 결과를 그리드 이미지로 저장한다."""
import os

# faiss + torch OpenMP 충돌 방지 (macOS)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import random
from pathlib import Path

import torch  # torch 먼저 — faiss OpenMP 충돌 방지
import faiss
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models import EfficientNetEmbedder
from src.utils import get_transforms

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_samples(data_root: Path) -> list[tuple[Path, str]]:
    """(이미지 경로, 캐릭터 라벨) 목록을 반환한다."""
    samples = []
    for char_dir in sorted(data_root.iterdir()):
        if char_dir.is_dir():
            for img in sorted(char_dir.iterdir()):
                if img.suffix.lower() in EXTS:
                    samples.append((img, char_dir.name))
    return samples


def encode_batch(paths: list[Path], model: EfficientNetEmbedder, device: torch.device, transform) -> np.ndarray:
    tensors = [transform(Image.open(p).convert("RGB")) for p in paths]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        return model.encode(batch).cpu().numpy()


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    embs = embeddings.copy()
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def draw_grid(
    query_path: Path,
    query_label: str,
    results: list[tuple[str, float, Path]],
    out_path: Path,
) -> None:
    """쿼리 1장 + top-k 결과를 한 행으로 시각화한다."""
    n_cols = 1 + len(results)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))

    def show(ax, path: Path, title: str, border_color: str | None = None) -> None:
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
        if border_color:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(border_color)
                spine.set_linewidth(5)

    show(axes[0], query_path, f"[Query]\n{query_label}")

    for i, (label, score, path) in enumerate(results):
        color = "#27ae60" if label == query_label else "#e74c3c"
        show(axes[i + 1], path, f"#{i + 1}  sim={score:.3f}\n{label}", color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS retrieval 결과를 이미지 그리드로 저장한다.")
    parser.add_argument("--checkpoint", required=True, help="학습된 체크포인트 경로")
    parser.add_argument("--data-root", default="data/processed", help="캐릭터 폴더 루트")
    parser.add_argument("--n-queries", type=int, default=5, help="시각화할 쿼리 이미지 수")
    parser.add_argument("--top-k", type=int, default=5, help="검색 결과 상위 K개")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/retrieval_demo")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_transforms(args.image_size, train=False)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = EfficientNetEmbedder(pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    samples = collect_samples(Path(args.data_root))
    if len(samples) < args.n_queries + 1:
        raise ValueError(
            f"데이터 부족: {len(samples)}개 이미지, 최소 {args.n_queries + 1}개 필요"
        )

    query_indices = set(random.sample(range(len(samples)), args.n_queries))
    queries = [samples[i] for i in sorted(query_indices)]
    gallery = [(p, l) for i, (p, l) in enumerate(samples) if i not in query_indices]
    gallery_paths, gallery_labels = zip(*gallery)

    # gallery 임베딩 빌드 (self-match 오염 방지를 위해 query 제외)
    print(f"Gallery 인코딩 중... ({len(gallery_paths)}장)")
    gallery_embs: list[np.ndarray] = []
    for i in range(0, len(gallery_paths), args.batch_size):
        batch = list(gallery_paths[i : i + args.batch_size])
        gallery_embs.append(encode_batch(batch, model, device, transform))
    index = build_faiss_index(np.vstack(gallery_embs))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    print(f"\n{'결과':─<60}")
    for q_path, q_label in queries:
        q_emb = encode_batch([q_path], model, device, transform)
        faiss.normalize_L2(q_emb)
        scores, idxs = index.search(q_emb, args.top_k)
        results = [
            (gallery_labels[i], float(scores[0][r]), gallery_paths[i])
            for r, i in enumerate(idxs[0])
        ]

        top1_ok = results[0][0] == q_label
        correct += int(top1_ok)
        mark = "✓" if top1_ok else "✗"
        print(f"{mark} Query: {q_label:30s} → Rank-1: {results[0][0]:30s} (sim={results[0][1]:.3f})")

        draw_grid(q_path, q_label, results, out_dir / f"{q_path.stem}.png")

    print(f"\nTop-1 Accuracy: {correct}/{len(queries)} = {correct / len(queries) * 100:.1f}%")
    print(f"결과 이미지 저장 위치: {out_dir}/")


if __name__ == "__main__":
    main()
