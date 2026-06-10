"""ArcFace loss와 CrossEntropy loss의 성능을 동일 backbone에서 비교한다.

backbone : EfficientNet-B0 (이전 비교 실험에서 최선의 EfficientNet)
평가 지표:
  - CE    : top-1 / top-5 분류 정확도
  - ArcFace : top-1 / top-5 분류 정확도 (코사인 유사도 분류기)
              Recall@1 / Recall@5 (FAISS 기반 retrieval)

출력:
  outputs/loss_comparison/results.csv
  outputs/loss_comparison/summary.csv
  outputs/loss_comparison/curves.png
  outputs/loss_comparison/summary_bar.png
"""
import time
from pathlib import Path

import faiss
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from tqdm import tqdm

from src.datasets import AnimeCharacterDataset
from src.losses import ArcFaceLoss
from src.models import EfficientNetEmbedder
from src.utils import get_transforms, seed_everything

matplotlib.rcParams["font.family"] = "AppleGothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 설정 ──────────────────────────────────────────────────────────────────────
BACKBONE    = "efficientnet_b0"
DATA_ROOT   = "data/processed"
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
PHASE1_EP   = 3
PHASE2_EP   = 7
LR_HEAD     = 1e-3
LR_FINETUNE = 1e-4
EMB_DIM     = 512
SEED        = 42
OUT_DIR     = Path("outputs/loss_comparison")
# ─────────────────────────────────────────────────────────────────────────────


def make_loaders(data_root: str, batch_size: int, seed: int):
    """train/val DataLoader를 각기 다른 transform으로 생성한다."""
    train_full = AnimeCharacterDataset(data_root, transform=get_transforms(IMAGE_SIZE, train=True), min_size=64)
    val_full   = AnimeCharacterDataset(data_root, transform=get_transforms(IMAGE_SIZE, train=False), min_size=64)

    n_train = int(len(train_full) * 0.8)
    n_val   = len(train_full) - n_train
    gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [s.indices for s in random_split(range(len(train_full)), [n_train, n_val], generator=gen)]

    weights = train_full.get_class_weights()[train_idx]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_idx), replacement=True)

    train_loader = DataLoader(Subset(train_full, train_idx), batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(Subset(val_full,   val_idx),   batch_size=batch_size, num_workers=2)

    # retrieval 평가용: train 임베딩 전체를 인덱스로 사용
    index_loader = DataLoader(Subset(val_full, train_idx), batch_size=batch_size, num_workers=2)

    return train_loader, val_loader, index_loader, train_full.classes, train_idx, val_idx


def freeze_backbone(model: nn.Module) -> None:
    classifier_names = {"head", "classifier", "fc"}
    for name, param in model.named_parameters():
        param.requires_grad = any(cn in name for cn in classifier_names)


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


# ── CrossEntropy 방식 ─────────────────────────────────────────────────────────

def train_ce(train_loader, val_loader, num_classes: int, device: torch.device) -> tuple[list[dict], nn.Module]:
    """CrossEntropy loss로 학습하고 epoch별 지표를 반환한다."""
    model = timm.create_model(BACKBONE, pretrained=True, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    records = []

    def run_epoch(training: bool, loader):
        model.train(training)
        total_loss, correct, total = 0.0, 0, 0
        ctx = torch.enable_grad() if training else torch.no_grad()
        opt = optimizer if training else None
        with ctx:
            for imgs, labels in tqdm(loader, desc="CE train" if training else "CE val", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                if opt:
                    opt.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, labels)
                if opt:
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        return total_loss / total, correct / total

    # Phase 1
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
    for ep in range(1, PHASE1_EP + 1):
        tr_loss, tr_acc = run_epoch(True,  train_loader)
        va_loss, va_acc = run_epoch(False, val_loader)
        records.append({"epoch": ep, "phase": 1, "train_loss": tr_loss, "val_loss": va_loss,
                         "train_acc": tr_acc, "val_acc": va_acc})
        print(f"  CE [{ep:02d}|P1] val_acc={va_acc:.4f}")

    # Phase 2
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EP)
    for ep in range(PHASE1_EP + 1, PHASE1_EP + PHASE2_EP + 1):
        tr_loss, tr_acc = run_epoch(True,  train_loader)
        va_loss, va_acc = run_epoch(False, val_loader)
        scheduler.step()
        records.append({"epoch": ep, "phase": 2, "train_loss": tr_loss, "val_loss": va_loss,
                         "train_acc": tr_acc, "val_acc": va_acc})
        print(f"  CE [{ep:02d}|P2] val_acc={va_acc:.4f}")

    return records, model


# ── ArcFace 방식 ──────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model: EfficientNetEmbedder, loader: DataLoader, device: torch.device):
    """모델로 임베딩을 추출하고 (embeddings, labels) 배열을 반환한다."""
    model.eval()
    embs, lbls = [], []
    for imgs, labels in tqdm(loader, desc="embed", leave=False):
        imgs = imgs.to(device)
        e = model(imgs)
        e = F.normalize(e, dim=-1)
        embs.append(e.cpu().numpy())
        lbls.extend(labels.numpy())
    return np.vstack(embs), np.array(lbls)



def eval_arcface_acc(model: EfficientNetEmbedder, criterion: ArcFaceLoss,
                     val_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """ArcFace weight matrix를 코사인 유사도 분류기로 사용해 top-1/top-5 정확도를 계산한다."""
    model.eval()
    criterion.eval()
    correct1, correct5, total = 0, 0, 0
    weight = F.normalize(criterion.weight, dim=-1)
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            emb = F.normalize(model(imgs), dim=-1)
            logits = emb @ weight.T
            correct1 += (logits.argmax(1) == labels).sum().item()
            correct5 += sum(labels[i] in logits[i].topk(5).indices for i in range(len(labels)))
            total += len(labels)
    return correct1 / total, correct5 / total


def train_arcface(train_loader, val_loader, index_loader, num_classes: int,
                  device: torch.device) -> tuple[list[dict], EfficientNetEmbedder, ArcFaceLoss]:
    """ArcFace loss로 학습하고 epoch별 지표를 반환한다."""
    model     = EfficientNetEmbedder(BACKBONE, EMB_DIM, pretrained=True).to(device)
    criterion = ArcFaceLoss(EMB_DIM, num_classes).to(device)
    records   = []

    def run_train_epoch(opt):
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc="AF train", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    # Phase 1: projector만 학습
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion.parameters()),
        lr=LR_HEAD,
    )
    for ep in range(1, PHASE1_EP + 1):
        tr_loss = run_train_epoch(optimizer)
        top1, top5 = eval_arcface_acc(model, criterion, val_loader, device)
        records.append({"epoch": ep, "phase": 1, "train_loss": tr_loss,
                         "val_acc_top1": top1, "val_acc_top5": top5})
        print(f"  AF [{ep:02d}|P1] top1={top1:.4f}")

    # Phase 2: 전체 fine-tune
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()), lr=LR_FINETUNE, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EP)
    for ep in range(PHASE1_EP + 1, PHASE1_EP + PHASE2_EP + 1):
        tr_loss = run_train_epoch(optimizer)
        top1, top5 = eval_arcface_acc(model, criterion, val_loader, device)
        scheduler.step()
        records.append({"epoch": ep, "phase": 2, "train_loss": tr_loss,
                         "val_acc_top1": top1, "val_acc_top5": top5})
        print(f"  AF [{ep:02d}|P2] top1={top1:.4f}")

    return records, model, criterion


def plot_curves(ce_records, af_records, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ce_df = pd.DataFrame(ce_records)
    af_df = pd.DataFrame(af_records)

    axes[0].plot(ce_df["epoch"], ce_df["train_loss"], label="CE train")
    axes[0].plot(af_df["epoch"], af_df["train_loss"], label="ArcFace train")
    axes[0].axvline(PHASE1_EP + 0.5, color="gray", linestyle="--", alpha=0.5, label="fine-tune 시작")
    axes[0].set(title="학습 손실 (Train Loss)", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(ce_df["epoch"], ce_df["val_acc"],      label="CE top-1")
    axes[1].plot(af_df["epoch"], af_df["val_acc_top1"], label="ArcFace top-1")
    axes[1].plot(af_df["epoch"], af_df["val_acc_top5"], label="ArcFace top-5", linestyle="--")
    axes[1].axvline(PHASE1_EP + 0.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].set(title="검증 정확도 (Val Accuracy)", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "curves.png", dpi=150)
    plt.close()


def plot_summary(summary: dict, out_dir: Path) -> None:
    labels  = ["CE top-1", "ArcFace top-1", "ArcFace top-5", "ArcFace Recall@1", "ArcFace Recall@5"]
    values  = [
        summary["ce_top1"], summary["af_top1"], summary["af_top5"],
        summary["af_recall1"], summary["af_recall5"],
    ]
    colors = ["steelblue", "tomato", "tomato", "tomato", "tomato"]
    alphas = [1.0, 1.0, 0.6, 1.0, 0.6]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.set(title="CE vs ArcFace 최종 성능 비교", xlabel="Accuracy / Recall")
    ax.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()


def main() -> None:
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}  |  Backbone: {BACKBONE}\n")

    train_loader, val_loader, index_loader, classes, train_idx, val_idx = make_loaders(DATA_ROOT, BATCH_SIZE, SEED)
    num_classes = len(classes)
    print(f"Classes: {num_classes}  |  Train: {len(train_idx)}  |  Val: {len(val_idx)}\n")

    # ── CE 학습 ───────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  [1/2] CrossEntropy Loss")
    print("=" * 50)
    t0 = time.time()
    ce_records, ce_model = train_ce(train_loader, val_loader, num_classes, device)
    ce_time = time.time() - t0
    ce_top1 = max(r["val_acc"] for r in ce_records)
    ce_top5_records = ce_records  # top-5 미측정 — 필요 시 추가 가능

    # ── ArcFace 학습 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  [2/2] ArcFace Loss")
    print("=" * 50)
    t0 = time.time()
    af_records, af_model, af_criterion = train_arcface(train_loader, val_loader, index_loader, num_classes, device)
    af_time = time.time() - t0
    af_top1 = max(r["val_acc_top1"] for r in af_records)
    af_top5 = max(r["val_acc_top5"] for r in af_records)

    # ── Retrieval 평가 ────────────────────────────────────────────────────────
    print("\nRetrieval 평가 중...")
    val_ds_no_aug = AnimeCharacterDataset(DATA_ROOT, transform=get_transforms(IMAGE_SIZE, train=False), min_size=64)
    index_loader2 = DataLoader(Subset(val_ds_no_aug, train_idx), batch_size=BATCH_SIZE, num_workers=2)
    query_loader  = DataLoader(Subset(val_ds_no_aug, val_idx),   batch_size=BATCH_SIZE, num_workers=2)

    index_embs, index_labels = extract_embeddings(af_model, index_loader2, device)
    query_embs, query_labels = extract_embeddings(af_model, query_loader,  device)

    af_recall1 = _recall_batch(index_embs, index_labels, query_embs, query_labels, k=1)
    af_recall5 = _recall_batch(index_embs, index_labels, query_embs, query_labels, k=5)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    summary = {
        "ce_top1": ce_top1, "ce_time": ce_time,
        "af_top1": af_top1, "af_top5": af_top5,
        "af_recall1": af_recall1, "af_recall5": af_recall5, "af_time": af_time,
    }

    print("\n" + "=" * 60)
    print("  최종 결과")
    print("=" * 60)
    print(f"  CE       top-1 acc : {ce_top1:.4f}  ({ce_time:.0f}s)")
    print(f"  ArcFace  top-1 acc : {af_top1:.4f}  top-5: {af_top5:.4f}  ({af_time:.0f}s)")
    print(f"  ArcFace  Recall@1  : {af_recall1:.4f}  Recall@5: {af_recall5:.4f}")

    pd.DataFrame(ce_records).assign(loss_type="CE").to_csv(OUT_DIR / "ce_records.csv", index=False)
    pd.DataFrame(af_records).assign(loss_type="ArcFace").to_csv(OUT_DIR / "af_records.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "summary.csv", index=False)

    plot_curves(ce_records, af_records, OUT_DIR)
    plot_summary(summary, OUT_DIR)
    print(f"\n산출물 저장 완료 → {OUT_DIR}/")


def _recall_batch(index_embs, index_labels, query_embs, query_labels, k: int) -> float:
    """FAISS IndexFlatIP로 Recall@k를 계산한다."""
    idx = faiss.IndexFlatIP(index_embs.shape[1])
    normed_index = index_embs.copy()
    faiss.normalize_L2(normed_index)
    idx.add(normed_index)

    normed_query = query_embs.copy()
    faiss.normalize_L2(normed_query)
    _, I = idx.search(normed_query, k)

    correct = sum(
        query_labels[qi] in index_labels[neighbors]
        for qi, neighbors in enumerate(I)
    )
    return correct / len(query_labels)


if __name__ == "__main__":
    main()
