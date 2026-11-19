"""ViT-B/16 encoder and the three panoramic instance objectives."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from ..config import InstanceConfig, VisionConfig
from ..data import MEAN, STD
from ..runtime import loss_mean


class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.patch_size <= 0 or config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.config = config
        self.patch_count = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, config.width, config.patch_size, stride=config.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.empty(1, self.patch_count + 1, config.width))
        nn.init.trunc_normal_(self.position, std=0.02)
        block = nn.TransformerEncoderLayer(
            config.width, config.heads, config.width * 4, dropout=0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(
            block, config.depth, norm=nn.LayerNorm(config.width, eps=1e-6),
            enable_nested_tensor=False,
        )
        for layer in self.layers.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)

    def forward(self, image: torch.Tensor, mask: torch.Tensor | None = None,
                mask_token: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.patch_embed(image).flatten(2).transpose(1, 2)
        if mask is not None:
            tokens = torch.where(mask[..., None], mask_token.to(tokens), tokens)
        tokens = torch.cat((self.cls_token.to(tokens).expand(len(image), -1, -1), tokens), dim=1)
        result = self.layers(tokens + self.position.to(tokens))
        return result[:, 0], result[:, 1:]


def gather_features(features: torch.Tensor, metadata: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather variable numbers of teeth for the replicated global contrastive loss."""
    dist = torch.distributed
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return features, metadata
    world_size, rank = dist.get_world_size(), dist.get_rank()
    lengths = [torch.zeros((), device=features.device, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(lengths, torch.tensor(len(features), device=features.device))
    sizes = [int(length) for length in lengths]
    padding = max(sizes) - len(features)
    padded = F.pad(features.detach(), (0, 0, 0, padding))
    feature_parts = [torch.empty_like(padded) for _ in sizes]
    dist.all_gather(feature_parts, padded)
    # Every rank sees the same objective. Its local derivative is sufficient
    # once scaled to account for DDP's gradient average across ranks.
    feature_parts[rank] = features.detach() + world_size * (features - features.detach())
    metadata = F.pad(metadata, (0, 0, 0, padding))
    meta_parts = [torch.empty_like(metadata) for _ in sizes]
    dist.all_gather(meta_parts, metadata)
    return (
        torch.cat([part[:size] for part, size in zip(feature_parts, sizes)]),
        torch.cat([part[:size] for part, size in zip(meta_parts, sizes)]),
    )


def contrastive_loss(features: torch.Tensor, meta: torch.Tensor, temperature: float) -> torch.Tensor:
    arch, fdi, view = meta.T
    same_arch = arch[:, None] == arch
    same_fdi = fdi[:, None] == fdi
    quadrant = fdi // 10
    contralateral = ((quadrant[:, None] - 1) ^ 1) == quadrant - 1
    symmetric = contralateral & (fdi[:, None] % 10 == fdi % 10)
    positives = same_arch & ((same_fdi & (view[:, None] != view)) | symmetric)
    diagonal = torch.eye(len(features), dtype=torch.bool, device=features.device)
    normalized = F.normalize(features.float(), dim=-1)
    scores = (normalized @ normalized.T) / temperature
    log_denominator = scores.masked_fill(diagonal, -torch.inf).logsumexp(dim=1)
    log_numerator = scores.masked_fill(~positives, -torch.inf).logsumexp(dim=1)
    return (log_denominator - log_numerator).mean()


class InstancePretrainer(nn.Module):
    def __init__(self, encoder: VisionEncoder, config: InstanceConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        if not 0 < config.mask_ratio <= 1 or config.temperature <= 0:
            raise ValueError("Instance mask ratio or temperature is out of range")
        if not 0 <= config.ema_momentum < 1:
            raise ValueError("EMA momentum must be in [0, 1)")
        if min(config.contrastive_weight, config.position_weight, config.mim_weight) < 0 or not any(
            (config.contrastive_weight, config.position_weight, config.mim_weight)
        ):
            raise ValueError("Use at least one instance objective with a positive weight")
        width = encoder.config.width
        patch_size = encoder.config.patch_size
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 128))
        self.momentum_encoder = deepcopy(encoder).requires_grad_(False) if config.contrastive_weight else None
        self.position = nn.Linear(2 * width, 1)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.decoder = nn.Linear(width, 3 * patch_size**2)

    @torch.no_grad()
    def update_momentum(self, momentum: float) -> None:
        if self.momentum_encoder is not None:
            for key, query in zip(self.momentum_encoder.parameters(), self.encoder.parameters(), strict=True):
                key.lerp_(query, 1 - momentum)

    def forward(self, images: torch.Tensor, arch: torch.Tensor, fdi: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        losses = {}
        if cfg.contrastive_weight or cfg.position_weight:
            clean, _ = self.encoder(images)
        if cfg.contrastive_weight:
            # All crops are grayscale replicated across channels before standardization.
            gray = images[:, :1] * 0.229 + 0.485
            factor = torch.empty_like(gray[:, :, :1, :1]).uniform_(0.8, 1.2)
            offset = torch.empty_like(factor).uniform_(-0.08, 0.08)
            augmented = self.momentum_encoder(self._standardize(gray * factor + offset))[0]
            projected = self.project(torch.cat((clean, augmented)))
            zero = torch.zeros_like(arch)
            metadata = torch.stack((arch.repeat(2), fdi.repeat(2), torch.cat((zero, zero + 1))), dim=1)
            projected, metadata = gather_features(projected, metadata)
            losses["contrastive"] = cfg.contrastive_weight * contrastive_loss(projected, metadata, cfg.temperature)
        if cfg.position_weight:
            index = fdi % 10
            first, second = torch.where((arch[:, None] == arch) & (index[:, None] != index))
            pairs = torch.cat((clean[first], clean[second]), dim=-1)
            score = self.position(pairs).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(score, (index[first] < index[second]).float(), reduction="sum")
            losses["position"] = cfg.position_weight * loss_mean(loss, len(first))
        if cfg.mim_weight:
            count = self.encoder.patch_count
            n_masked = max(1, round(count * cfg.mask_ratio))
            selected = torch.rand(len(images), count, device=images.device).argsort(dim=1)[:, :n_masked]
            mask = torch.zeros(len(images), count, device=images.device, dtype=torch.bool).scatter_(1, selected, True)
            patches = self.encoder(images, mask, self.mask_token)[1]
            target = F.unfold(images, self.encoder.config.patch_size, stride=self.encoder.config.patch_size).transpose(1, 2)
            residual = self.decoder(patches[mask]).float() - target[mask].float()
            losses["mim"] = cfg.mim_weight * loss_mean(residual.square().sum(), len(residual))
        return losses

    @staticmethod
    def _standardize(gray: torch.Tensor) -> torch.Tensor:
        return (gray.clamp(0, 1) - MEAN.to(gray)) / STD.to(gray)
