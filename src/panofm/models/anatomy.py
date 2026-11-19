"""FDI anatomical priors and channel-level optimal transport."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..config import OTConfig


def sinkhorn(cost: torch.Tensor, regularization: float, iterations: int) -> torch.Tensor:
    """Solve the balanced uniform-marginal transport problem in log space."""
    log_kernel = -cost.float() / regularization
    rows, columns = log_kernel.shape
    row_mass, column_mass = -math.log(rows), -math.log(columns)
    v = log_kernel.new_zeros(columns)
    for _ in range(iterations):
        u = row_mass - torch.logsumexp(log_kernel + v[None, :], dim=1)
        v = column_mass - torch.logsumexp(log_kernel + u[:, None], dim=0)
    return (log_kernel + u[:, None] + v[None, :]).exp()


class AnatomicalAlignment(nn.Module):
    def __init__(self, visual_channels: int, patches: int, config: OTConfig):
        super().__init__()
        if not 1 <= config.clusters <= visual_channels or config.fit_samples < 1:
            raise ValueError("Invalid number of channel clusters or fit samples")
        if config.regularization <= 0 or config.iterations <= 0 or not 0 <= config.threshold <= 1:
            raise ValueError("Invalid transport regularization, iterations, or threshold")
        self.config = config
        # Dental arch, quadrant, laterality, mesial-distal order, and tooth group.
        self.embeddings = nn.ModuleList(nn.Embedding(n, patches) for n in (2, 4, 2, 8, 4))
        self.visual_projection = nn.Linear(patches, config.projection_width)
        self.anatomy_projection = nn.Linear(patches, config.projection_width)
        self.register_buffer("cluster_index", torch.arange(visual_channels) % config.clusters)

    @torch.no_grad()
    def fit_clusters(self, visual: torch.Tensor, seed: int) -> None:
        from sklearn.cluster import KMeans

        # A channel's profile is the patch activation vector across training teeth.
        profiles = visual.transpose(1, 2).transpose(0, 1).flatten(1).float()
        labels = KMeans(self.config.clusters, n_init=10, random_state=seed).fit_predict(profiles.cpu().numpy())
        self.cluster_index.copy_(torch.as_tensor(labels, device=self.cluster_index.device))

    def anatomy(self, fdi: torch.Tensor) -> torch.Tensor:
        quadrant, order = fdi // 10 - 1, fdi % 10 - 1
        left = (quadrant == 1) | (quadrant == 2)
        group = (order >= 2).long() + (order >= 3).long() + (order >= 5).long()
        values = (quadrant // 2, quadrant, left.long(), order, group)
        return torch.stack([embed(value) for embed, value in zip(self.embeddings, values)], dim=1)

    def cost(self, visual: torch.Tensor, fdi: torch.Tensor) -> torch.Tensor:
        assignment = F.one_hot(self.cluster_index, self.config.clusters).to(visual)
        clustered = torch.einsum("bpc,ck->bkp", visual, assignment)
        clustered = clustered / assignment.sum(dim=0).clamp_min(1)[None, :, None]
        anatomy = self.anatomy_projection(self.anatomy(fdi))
        pathology = self.visual_projection(clustered)
        return torch.cdist(anatomy.float(), pathology.float())

    def transport(self, cost: torch.Tensor) -> torch.Tensor:
        return sinkhorn(cost, self.config.regularization, self.config.iterations)

    def forward(self, visual: torch.Tensor, fdi: torch.Tensor) -> torch.Tensor:
        cost = self.cost(visual, fdi).mean(dim=0)
        dist = torch.distributed
        if self.training and dist.is_initialized() and dist.get_world_size() > 1:
            from torch.distributed.nn.functional import all_reduce

            cost = all_reduce(cost) / dist.get_world_size()
        plan = self.transport(cost)
        hard = plan >= self.config.threshold
        best = F.one_hot(plan.argmax(dim=1), self.config.clusters).bool()
        hard = hard | (~hard.any(dim=1, keepdim=True) & best)
        soft = plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-12)
        # Preserve the binary mask in the forward pass and train projections through the soft plan.
        mask = (hard.float() + (soft - soft.detach()))[:, self.cluster_index].to(visual)
        return torch.einsum("bpc,ac->bap", visual, mask)
