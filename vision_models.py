from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class VisionModelCfg:
    image_size: int = 84
    in_channels: int = 1
    features_dim: int = 128


class SpiralVisionBackbone(nn.Module):
    def __init__(self, cfg: VisionModelCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, cfg.in_channels, cfg.image_size, cfg.image_size)
            n_flat = int(self.conv(dummy).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(n_flat, cfg.features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


class BallRegressor(nn.Module):
    def __init__(self, cfg: VisionModelCfg, out_dim: int = 2) -> None:
        super().__init__()
        self.backbone = SpiralVisionBackbone(cfg)
        self.head = nn.Linear(cfg.features_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)
