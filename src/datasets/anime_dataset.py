from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class AnimeCharacterDataset(Dataset):
    """캐릭터별 폴더 구조를 ImageFolder 방식으로 읽는 데이터셋.

    기대하는 디렉터리 구조:
        root/
          <캐릭터명>/
            img001.jpg
            img002.jpg
            ...
    """

    def __init__(self, root: str, transform=None, min_size: int = 64):
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for img_path in (self.root / cls).iterdir():
                if img_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                # 너무 작은 이미지(썸네일·아이콘 등)는 학습에서 제외한다.
                with Image.open(img_path) as img:
                    w, h = img.size
                if w < min_size or h < min_size:
                    continue
                self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_weights(self) -> torch.Tensor:
        """WeightedRandomSampler용 샘플별 가중치를 반환한다.

        클래스 빈도의 역수를 사용해 소수 클래스를 더 자주 샘플링한다.
        """
        class_counts = torch.zeros(len(self.classes))
        for _, label in self.samples:
            class_counts[label] += 1
        weight_per_class = 1.0 / class_counts.clamp(min=1)
        return torch.tensor([weight_per_class[label] for _, label in self.samples])
