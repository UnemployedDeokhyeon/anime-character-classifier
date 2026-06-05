from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class AnimeCharacterDataset(Dataset):
    """캐릭터별 폴더 구조를 ImageFolder 방식으로 읽는 데이터셋.

    기대하는 디렉터리 구조:
        root/
          <캐릭터명>/
            img001.jpg
            img002.jpg
            ...
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for img_path in (self.root / cls).iterdir():
                # 학습 가능한 이미지 확장자만 샘플 목록에 등록한다.
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
