from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .common import WORLD_MODEL_SPECS, WorldModelSpec
from .models import ResidualBlock


class EventSnakeWorldModel(nn.Module):
    """Context-1 Snake world model with hard event outputs.

    Inputs are the previous RGB frame and the next discrete action. Outputs are
    the next RGB frame, apple-event logits, and death logits.
    """

    def __init__(self, variant: str = "wm_1m"):
        super().__init__()
        if variant not in WORLD_MODEL_SPECS:
            raise ValueError(f"unknown world model variant: {variant}")
        self.context = 1
        self.variant = variant
        spec = WORLD_MODEL_SPECS[variant]
        base = spec.base_channels
        input_ch = 3 + 4
        self.encoder = nn.Sequential(
            nn.Conv2d(input_ch, base, 5, stride=2, padding=2),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(4, base * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(4, base * 4),
            nn.SiLU(inplace=True),
        )
        self.dynamics = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(spec.latent_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(4, base * 2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(base, base, 4, stride=2, padding=1),
            nn.GroupNorm(4, base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 3, 3, padding=1),
            nn.Sigmoid(),
        )
        self.event_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(base * 4, base * 4),
            nn.SiLU(inplace=True),
            nn.Linear(base * 4, 4),
        )

    @property
    def spec(self) -> WorldModelSpec:
        return WORLD_MODEL_SPECS[self.variant]

    def build_input(self, frame: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if frame.ndim == 5:
            if frame.shape[1] != 1:
                raise ValueError(f"event world model expects context=1, got {tuple(frame.shape)}")
            frame = frame[:, 0]
        b, c, h, w = frame.shape
        if c != 3:
            raise ValueError(f"expected B,3,H,W frame, got {tuple(frame.shape)}")
        action_plane = F.one_hot(action.long(), num_classes=4).float().view(b, 4, 1, 1).expand(b, 4, h, w)
        return torch.cat([frame, action_plane], dim=1)

    def forward(self, frame: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.dynamics(self.encoder(self.build_input(frame, action)))
        event = self.event_head(z)
        return {
            "frame": self.decoder(z),
            "apple_logits": event[:, :2],
            "death_logits": event[:, 2:],
            "latent": z,
        }

    def config(self) -> dict[str, Any]:
        return {"context": 1, "variant": self.variant, "event_model": True, "spec": asdict(self.spec)}

