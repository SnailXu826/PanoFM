"""DINOv2 guidance and Lyapunov-controlled disease prototypes."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..config import PrototypeConfig
from ..data import DISEASES
from .vision import VisionEncoder


class DinoTeacher(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.classifier = nn.Linear(model.config.hidden_size, len(DISEASES))

    @classmethod
    def from_local(cls, path: Path):
        from transformers import Dinov2Model

        if not path.is_dir():
            raise ValueError("Provide a local DINOv2 model directory")
        return cls(Dinov2Model.from_pretrained(path, local_files_only=True))

    @classmethod
    def from_config(cls, config: dict):
        from transformers import Dinov2Config, Dinov2Model

        return cls(Dinov2Model(Dinov2Config.from_dict(config)))

    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=images).last_hidden_state[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class PrototypeBank(nn.Module):
    def __init__(self, classes: int, width: int, config: PrototypeConfig):
        super().__init__()
        if not 0 < config.update_lr <= 0.1 or min(config.alpha, config.beta) < 0:
            raise ValueError("Invalid prototype learning rate or loss weights")
        self.config = config
        self.register_buffer("centroids", torch.zeros(classes, width))
        self.register_buffer("rate", torch.tensor(config.update_lr, dtype=torch.float64))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor) -> bool:
        # Evaluate both candidates against the same, newly encoded batch.
        centroids = self.centroids.double()
        features = features.double()
        residual = centroids[labels] - features
        gradient = torch.zeros_like(centroids).index_add_(0, labels, residual)
        count = torch.tensor(len(labels), device=features.device, dtype=torch.float64)
        dist = torch.distributed
        if dist.is_initialized():
            dist.all_reduce(gradient)
            dist.all_reduce(count)
        candidate = (centroids - 2 * self.rate * gradient / count).to(self.centroids)
        candidate_error = (candidate[labels].double() - features).square().sum()
        delta = candidate_error - residual.square().sum()
        if dist.is_initialized():
            dist.all_reduce(delta)
        accepted = bool(torch.isfinite(delta)) and (not self.config.stability or bool(delta / count <= 1e-9))
        if accepted:
            self.centroids.copy_(candidate)
        if self.config.stability:
            self.rate.mul_(1.01 if accepted else 0.5).clamp_(max=0.1)
        return accepted


class PrototypePretrainer(nn.Module):
    def __init__(self, encoder: VisionEncoder, config: PrototypeConfig):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.width, len(DISEASES))
        self.bank = PrototypeBank(len(DISEASES), encoder.config.width, config)

    def forward(self, images: torch.Tensor, labels: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)[0]
        classification = F.cross_entropy(self.classifier(features).float(), labels)
        teacher_distances = (features.float() - reference.float()).square().sum()
        prototype_distances = (features.float() - self.bank.centroids[labels]).square().sum()
        dist = torch.distributed
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        alignment = self.bank.config.alpha * teacher_distances + self.bank.config.beta * prototype_distances
        return classification + world_size * alignment
