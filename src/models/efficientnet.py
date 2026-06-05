import timm
import torch
import torch.nn as nn


class EfficientNetEmbedder(nn.Module):
    """EfficientNet backbone with a projection head for metric learning.

    Based on AniWho (arXiv:2208.11012) which found EfficientNet-B7 achieves
    85.08% top-1 accuracy on anime character face classification.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b7",
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        """Initialize the backbone and projection head."""
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
        """Compute embeddings for a batch of images."""
        features = self.backbone(x)
        return self.projector(features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings without gradient tracking."""
        with torch.no_grad():
            emb = self.forward(x)
            return nn.functional.normalize(emb, dim=-1)
