from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """학습/검증 루프와 체크포인트 저장을 담당한다."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.scaler = GradScaler(enabled=use_amp)

    def train_epoch(self, loader: DataLoader) -> float:
        """한 epoch 동안 학습하고 평균 loss를 반환한다."""
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="train"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.scaler.is_enabled()):
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
            # AMP가 꺼져 있으면 GradScaler는 일반 backward/step처럼 동작한다.
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        """검증 데이터셋의 평균 loss를 계산한다."""
        self.model.eval()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="val"):
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings = self.model(images)
            loss = self.criterion(embeddings, labels)
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """현재 epoch의 모델/옵티마이저 상태를 저장한다."""
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.checkpoint_dir / f"epoch_{epoch:03d}_loss{val_loss:.4f}.pt",
        )
