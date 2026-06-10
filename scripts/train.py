"""애니 캐릭터 임베딩 모델을 학습한다."""
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from src.datasets import AnimeCharacterDataset
from src.losses import ArcFaceLoss
from src.models import EfficientNetEmbedder
from src.trainers import Trainer
from src.utils import get_transforms, seed_everything


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra 설정을 읽어 데이터셋, 모델, 손실 함수, 학습 루프를 구성한다."""
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train/val에 서로 다른 transform을 적용하기 위해 같은 root를 두 번 로드한다.
    # random_split을 동일 seed로 실행하면 인덱스가 일치해 데이터 누수가 없다.
    train_full = AnimeCharacterDataset(
        cfg.data.root,
        transform=get_transforms(cfg.data.image_size, train=True, random_erasing_p=cfg.data.random_erasing_p),
        min_size=cfg.data.min_image_size,
    )
    val_full = AnimeCharacterDataset(
        cfg.data.root,
        transform=get_transforms(cfg.data.image_size, train=False),
        min_size=cfg.data.min_image_size,
    )

    n_train = int(len(train_full) * cfg.data.train_split)
    n_val = len(train_full) - n_train
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = [s.indices for s in random_split(range(len(train_full)), [n_train, n_val], generator=generator)]

    train_ds = Subset(train_full, train_indices)
    val_ds = Subset(val_full, val_indices)

    weights = train_full.get_class_weights()[train_indices]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, sampler=sampler, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers)

    model = EfficientNetEmbedder(
        backbone=cfg.model.backbone,
        embedding_dim=cfg.model.embedding_dim,
        pretrained=cfg.model.pretrained,
    )
    criterion = ArcFaceLoss(
        embedding_dim=cfg.model.embedding_dim,
        num_classes=len(dataset.classes),
        scale=cfg.loss.scale,
        margin=cfg.loss.margin,
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    trainer = Trainer(model, criterion, optimizer, scheduler, device, cfg.checkpoint.dir, cfg.training.amp)

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.eval_epoch(val_loader)
        scheduler.step()
        print(f"[{epoch:03d}] train={train_loss:.4f}  val={val_loss:.4f}")
        trainer.save_checkpoint(epoch, val_loss)


if __name__ == "__main__":
    main()
