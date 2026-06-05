"""5개 backbone의 전이 학습 성능을 비교한다.

전략:
  Phase 1 (5 epochs)  — backbone을 고정하고 classifier head만 학습
  Phase 2 (15 epochs) — 낮은 learning rate로 전체 fine-tune

출력:
  outputs/comparison/results.csv        — epoch별 지표
  outputs/comparison/summary.csv        — 최종 비교표
  outputs/comparison/loss_curve.png
  outputs/comparison/acc_curve.png
  outputs/comparison/summary_bar.png
"""
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets import AnimeCharacterDataset
from src.utils import get_transforms, seed_everything

matplotlib.rcParams["font.family"] = "Nanum Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 설정 ─────────────────────────────────────────────────────────────────────
MODELS = [
    ("EfficientNet-B7", "efficientnet_b7"),
    ("EfficientNet-B0", "efficientnet_b0"),
    ("MobileNetV2",     "mobilenetv2_100"),
    ("VGG19",           "vgg19"),
    ("ViT-B/16",        "vit_base_patch16_224"),
]
DATA_ROOT   = "data/processed"
IMAGE_SIZE  = 224
BATCH_SIZE  = 32
PHASE1_EP   = 5     # backbone 고정 학습
PHASE2_EP   = 15    # 전체 fine-tune
LR_HEAD     = 1e-3
LR_FINETUNE = 1e-4
OUT_DIR     = Path("outputs/comparison")
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────


def build_model(backbone: str, num_classes: int) -> nn.Module:
    """timm backbone 이름으로 분류 모델을 생성한다."""
    model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """classifier head를 제외한 모든 파라미터를 고정한다."""
    classifier_names = {"head", "classifier", "fc", "pre_logits"}
    for name, param in model.named_parameters():
        is_head = any(cn in name for cn in classifier_names)
        param.requires_grad = is_head


def unfreeze_all(model: nn.Module) -> None:
    """모든 파라미터를 다시 학습 가능하게 만든다."""
    for param in model.parameters():
        param.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """학습 또는 검증 1 epoch를 실행하고 평균 loss와 accuracy를 반환한다."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def train_one_model(
    name: str,
    backbone: str,
    train_ds,
    val_ds,
    num_classes: int,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    """단일 backbone에 대해 head 학습과 전체 fine-tune을 순서대로 수행한다."""
    model = build_model(backbone, num_classes).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    criterion    = nn.CrossEntropyLoss()
    records      = []
    t0           = time.time()

    # ── Phase 1: head만 학습 ─────────────────────────────────────────────────
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )
    for ep in range(1, PHASE1_EP + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device)
        records.append({"model": name, "epoch": ep, "phase": 1,
                        "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc,  "val_acc": va_acc})
        print(f"  [{ep:02d}|P1] loss={va_loss:.4f}  acc={va_acc:.4f}")

    # ── Phase 2: 전체 fine-tune ──────────────────────────────────────────────
    unfreeze_all(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EP)
    for ep in range(PHASE1_EP + 1, PHASE1_EP + PHASE2_EP + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device)
        scheduler.step()
        records.append({"model": name, "epoch": ep, "phase": 2,
                        "train_loss": tr_loss, "val_loss": va_loss,
                        "train_acc": tr_acc,  "val_acc": va_acc})
        print(f"  [{ep:02d}|P2] loss={va_loss:.4f}  acc={va_acc:.4f}")

    elapsed = time.time() - t0
    best_val_acc = max(r["val_acc"] for r in records)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    summary = {
        "model": name, "backbone": backbone,
        "best_val_acc": best_val_acc,
        "final_val_loss": records[-1]["val_loss"],
        "train_time_sec": round(elapsed, 1),
        "params_M": round(n_params, 1),
    }
    return pd.DataFrame(records), summary


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """모델별 검증 loss/accuracy 곡선을 저장한다."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    total_ep = PHASE1_EP + PHASE2_EP

    for model_name, grp in df.groupby("model"):
        x = grp["epoch"].tolist()
        axes[0].plot(x, grp["val_loss"],  label=model_name)
        axes[1].plot(x, grp["val_acc"],   label=model_name)

    for ax, title, ylabel in zip(
        axes,
        ["검증 손실 (Val Loss)", "검증 정확도 (Val Accuracy)"],
        ["Loss", "Accuracy"],
    ):
        ax.axvline(x=PHASE1_EP + 0.5, color="gray", linestyle="--", alpha=0.5, label="fine-tune 시작")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_dir / "curves.png", dpi=150)
    plt.close()
    print(f"Saved curves → {out_dir / 'curves.png'}")


def plot_summary(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """최종 성능, 학습 시간, 파라미터 수 비교 그래프를 저장한다."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, col, title, fmt in zip(
        axes,
        ["best_val_acc", "train_time_sec", "params_M"],
        ["최고 검증 정확도", "학습 시간 (초)", "파라미터 수 (M)"],
        [".3f", ".0f", ".1f"],
    ):
        bars = ax.barh(summary_df["model"], summary_df[col])
        ax.set_title(title)
        for bar, val in zip(bars, summary_df[col]):
            ax.text(bar.get_width() * 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:{fmt}}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    plt.close()
    print(f"Saved summary → {out_dir / 'summary.png'}")


def main() -> None:
    """비교 실험 전체를 실행하고 CSV/그래프 산출물을 저장한다."""
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    dataset = AnimeCharacterDataset(DATA_ROOT, transform=get_transforms(IMAGE_SIZE, train=True))
    num_classes = len(dataset.classes)
    n_train = int(len(dataset) * 0.8)
    train_ds, val_ds = random_split(dataset, [n_train, len(dataset) - n_train],
                                    generator=torch.Generator().manual_seed(SEED))
    val_ds.dataset.transform = get_transforms(IMAGE_SIZE, train=False)
    print(f"Classes: {num_classes}  |  Train: {n_train}  |  Val: {len(dataset) - n_train}\n")

    all_records: list[pd.DataFrame] = []
    summaries:   list[dict]         = []

    for name, backbone in MODELS:
        print(f"\n{'='*50}")
        print(f"  {name}  ({backbone})")
        print(f"{'='*50}")
        records_df, summary = train_one_model(name, backbone, train_ds, val_ds, num_classes, device)
        all_records.append(records_df)
        summaries.append(summary)

    results_df  = pd.concat(all_records, ignore_index=True)
    summary_df  = pd.DataFrame(summaries).sort_values("best_val_acc", ascending=False)

    results_df.to_csv(OUT_DIR / "results.csv",  index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv",  index=False)

    print("\n\n" + "="*60)
    print("  최종 비교 결과")
    print("="*60)
    print(summary_df.to_string(index=False))

    plot_curves(results_df, OUT_DIR)
    plot_summary(summary_df, OUT_DIR)


if __name__ == "__main__":
    main()
