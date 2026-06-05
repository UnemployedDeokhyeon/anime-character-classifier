from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class AnimeCharacterDataset(Dataset):
    """ImageFolder-style dataset for anime character images.

    Expected layout:
        root/
          <character_name>/
            img001.jpg
            img002.jpg
            ...
    """

    def __init__(self, root: str, transform=None):
        """Index images under the root directory."""
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for img_path in (self.root / cls).iterdir():
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and return a transformed image tensor and label index."""
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
