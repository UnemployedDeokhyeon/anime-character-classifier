from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Training helper that runs epochs and manages checkpoints."""

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
        """Store training components and AMP configuration."""
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.scaler = GradScaler(enabled=use_amp)

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="train"):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=self.scaler.is_enabled()):
                embeddings = self.model(images)
                loss = self.criterion(embeddings, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch and return average loss."""
        self.model.eval()
        total_loss = 0.0
        for images, labels in tqdm(loader, desc="val"):
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings = self.model(images)
            loss = self.criterion(embeddings, labels)
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Persist model/optimizer state for the given epoch."""
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "val_loss": val_loss,
            },
            self.checkpoint_dir / f"epoch_{epoch:03d}_loss{val_loss:.4f}.pt",
        )
