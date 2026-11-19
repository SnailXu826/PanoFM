"""Load the shared experiment configuration."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    width: int = 768
    depth: int = 12
    heads: int = 12


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    lr: float = 1.5e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.03
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    seed: int = 42


@dataclass(frozen=True)
class InstanceConfig:
    batch_size: int = 256
    mask_ratio: float = 0.7
    temperature: float = 0.1
    ema_momentum: float = 0.996
    contrastive_weight: float = 1.0
    position_weight: float = 1.0
    mim_weight: float = 1.0


@dataclass(frozen=True)
class PrototypeConfig:
    batch_size: int = 128
    alpha: float = 1e-4
    beta: float = 1e-4
    update_lr: float = 0.01
    stability: bool = True


@dataclass(frozen=True)
class DiagnosisConfig:
    batch_size: int = 128
    fusion_width: int = 256
    fusion_heads: int = 8
    fusion_depth: int = 2
    queries: int = 8
    text_length: int = 256


@dataclass(frozen=True)
class OTConfig:
    clusters: int = 16
    projection_width: int = 64
    iterations: int = 50
    regularization: float = 0.1
    threshold: float = 0.05
    fit_samples: int = 128


@dataclass(frozen=True)
class Config:
    vision: VisionConfig = field(default_factory=VisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    instance: InstanceConfig = field(default_factory=InstanceConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    diagnosis: DiagnosisConfig = field(default_factory=DiagnosisConfig)
    ot: OTConfig = field(default_factory=OTConfig)


def load_config(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as file:
        values = tomllib.load(file)
    defaults = Config()
    sections = {name: type(getattr(defaults, name))(**options) for name, options in values.items()}
    return Config(**sections)
