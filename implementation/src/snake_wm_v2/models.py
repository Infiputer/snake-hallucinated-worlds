from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .common import POLICY_SPECS, WORLD_MODEL_SPECS, PolicySpec, WorldModelSpec


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class LatentSnakeWorldModel(nn.Module):
    def __init__(self, context: int, variant: str = "wm_1m"):
        super().__init__()
        if variant not in WORLD_MODEL_SPECS:
            raise ValueError(f"unknown world model variant: {variant}")
        self.context = int(context)
        self.variant = variant
        spec = WORLD_MODEL_SPECS[variant]
        base = spec.base_channels
        input_ch = self.context * 3 + 1 + 4
        self.encoder = nn.Sequential(nn.Conv2d(input_ch, base, 5, stride=2, padding=2), nn.GroupNorm(4, base), nn.SiLU(inplace=True), nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(4, base * 2), nn.SiLU(inplace=True), nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.GroupNorm(4, base * 4), nn.SiLU(inplace=True))
        self.dynamics = nn.Sequential(*[ResidualBlock(base * 4) for _ in range(spec.latent_blocks)])
        self.decoder = nn.Sequential(nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.GroupNorm(4, base * 2), nn.SiLU(inplace=True), nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.GroupNorm(4, base), nn.SiLU(inplace=True), nn.ConvTranspose2d(base, base, 4, stride=2, padding=1), nn.GroupNorm(4, base), nn.SiLU(inplace=True), nn.Conv2d(base, 3, 3, padding=1), nn.Sigmoid())
        self.scalar_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(base * 4, base * 4), nn.SiLU(inplace=True), nn.Linear(base * 4, 3))

    @property
    def spec(self) -> WorldModelSpec:
        return WORLD_MODEL_SPECS[self.variant]

    def build_input(self, frames: torch.Tensor, prev_reward: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        b, ctx, c, h, w = frames.shape
        if ctx != self.context or c != 3:
            raise ValueError(f"expected B,{self.context},3,H,W, got {tuple(frames.shape)}")
        action_plane = F.one_hot(action.long(), num_classes=4).float().view(b, 4, 1, 1).expand(b, 4, h, w)
        reward_plane = prev_reward.float().view(b, 1, 1, 1).expand(b, 1, h, w)
        return torch.cat([frames.reshape(b, ctx * c, h, w), reward_plane, action_plane], dim=1)

    def forward(self, frames: torch.Tensor, prev_reward: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.dynamics(self.encoder(self.build_input(frames, prev_reward, action)))
        scalar = self.scalar_head(z)
        return {"frame": self.decoder(z), "reward": scalar[:, 0], "done_logit": scalar[:, 1], "length": scalar[:, 2], "latent": z}

    def config(self) -> dict[str, Any]:
        return {"context": self.context, "variant": self.variant, "spec": asdict(self.spec)}


class CNNPolicy(nn.Module):
    def __init__(self, variant: str = "small", action_dim: int = 4):
        super().__init__()
        if variant not in POLICY_SPECS:
            raise ValueError(f"unknown policy variant: {variant}")
        self.variant = variant
        spec = POLICY_SPECS[variant]
        c1, c2, c3 = spec.channels
        self.features = nn.Sequential(nn.Conv2d(3, c1, 5, stride=2, padding=2), nn.GroupNorm(4, c1), nn.SiLU(inplace=True), nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.GroupNorm(4, c2), nn.SiLU(inplace=True), nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.GroupNorm(4, c3), nn.SiLU(inplace=True), ResidualBlock(c3), nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        self.policy = nn.Sequential(nn.Linear(c3, spec.hidden), nn.SiLU(inplace=True), nn.Linear(spec.hidden, action_dim))
        self.value = nn.Sequential(nn.Linear(c3, spec.hidden), nn.SiLU(inplace=True), nn.Linear(spec.hidden, 1))

    @property
    def spec(self) -> PolicySpec:
        return POLICY_SPECS[self.variant]

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(frame)
        return self.policy(feat), self.value(feat).squeeze(-1)
