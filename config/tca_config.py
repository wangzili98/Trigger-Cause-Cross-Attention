"""TCA network configuration and minimum-example training controls."""
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
VERSION = '1.0.0'
MODEL_NAME = 'Trigger–Cause Cross-Attention'
MODEL_ABBREVIATION = 'TCA'
SOFTWARE_TITLE = f'{MODEL_NAME} ({MODEL_ABBREVIATION})'
SCHEMA = 'tca-real-sample-v1'


@dataclass(frozen=True)
class TCAConfig:
    cause_frames: int = 48
    trigger_frames: int = 18
    cause_channels: int = 3
    trigger_channels: int = 3
    environment_channels: int = 16
    cause_size: int = 64
    trigger_size: int = 16
    embed_dim: int = 256
    temporal_layers: int = 6
    heads: int = 8
    cause_cnn_depth: int = 4
    trigger_cnn_depth: int = 2
    branch_dropout: float = 0.1
    cross_dropout: float = 0.2
    head_hidden: int = 128
    head_dropout: float = 0.4
    positive_prior: float = 0.15

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    # Numerical choices absent from the manuscript are retained from supplied code.
    # These are demo values, not a claim of the paper's selected CV optimum.
    focal_alpha: float = 0.9
    focal_gamma: float = 2.0
    regression_weight: float = 0.1
    cause_lr: float = 2e-6
    trigger_lr: float = 5e-6
    fusion_lr: float = 3e-5
    head_lr: float = 5e-5
    branch_weight_decay: float = 0.05
    other_weight_decay: float = 0.005
    minimum_lr: float = 1e-6
    gradient_clip: float = 5.0
    pretrain_lr: float = 1e-4
    pretrain_weight_decay: float = 0.01
    mask_ratio: float = 0.3
    cause_patch: int = 16
    trigger_patch: int = 4
    seed: int = 42

    def to_dict(self):
        return asdict(self)
