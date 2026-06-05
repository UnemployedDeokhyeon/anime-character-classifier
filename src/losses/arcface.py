import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """ArcFace additive angular margin loss for metric learning."""

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss for a batch of embeddings and labels."""
        embeddings = F.normalize(embeddings, dim=-1)
        weight = F.normalize(self.weight, dim=-1)
        cos_theta = F.linear(embeddings, weight)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(0, 1))
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(cos_theta > self.th, cos_theta_m, cos_theta - self.mm)

        one_hot = torch.zeros_like(cos_theta).scatter_(1, labels.view(-1, 1), 1)
        logits = torch.where(one_hot.bool(), cos_theta_m, cos_theta) * self.scale
        return F.cross_entropy(logits, labels)
