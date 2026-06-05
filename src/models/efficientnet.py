import timm
import torch
import torch.nn as nn


class EfficientNetEmbedder(nn.Module):
    """EfficientNet backbone에 메트릭 러닝용 projection head를 붙인 모델.

    AniWho(arXiv:2208.11012)에서 애니 캐릭터 얼굴 분류 top-1 85.08%를
    기록한 EfficientNet-B7을 기본 backbone으로 사용한다.
    """

    def __init__(self, backbone: str = "efficientnet_b7", embedding_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        self.projector = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.BatchNorm1d(in_features // 2),
            nn.GELU(),
            nn.Linear(in_features // 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """입력 이미지 배치를 임베딩 벡터로 변환한다."""
        features = self.backbone(x)
        return self.projector(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """추론용 L2 정규화 임베딩을 반환한다."""
        with torch.no_grad():
            emb = self.forward(x)
            return nn.functional.normalize(emb, dim=-1)
