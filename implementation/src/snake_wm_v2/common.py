from __future__ import annotations

import json
import netrc
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


CONTEXTS = (1, 2, 5)
ACTIONS = (0, 1, 2, 3)


@dataclass(frozen=True)
class RewardConfig:
    apple_reward: float = 1.0
    death_penalty: float = -1.0
    win_reward: float = 5.0


@dataclass(frozen=True)
class WorldModelSpec:
    name: str
    base_channels: int
    latent_blocks: int


@dataclass(frozen=True)
class PolicySpec:
    name: str
    channels: tuple[int, int, int]
    hidden: int


WORLD_MODEL_SPECS = {
    "tiny": WorldModelSpec("tiny", 8, 1),
    "wm_1m": WorldModelSpec("wm_1m", 36, 2),
    "wm_2m": WorldModelSpec("wm_2m", 44, 3),
    "wm_5m": WorldModelSpec("wm_5m", 52, 6),
}

POLICY_SPECS = {
    "small": PolicySpec("small", (16, 32, 64), 128),
    "medium": PolicySpec("medium", (48, 96, 192), 384),
    "large": PolicySpec("large", (96, 192, 384), 768),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def shaped_reward(base_reward: float, status: str, cfg: RewardConfig) -> float:
    reward = cfg.apple_reward if base_reward > 0 else 0.0
    if status == "dead":
        reward += cfg.death_penalty
    elif status == "win":
        reward += cfg.win_reward
    return float(reward)


def resolve_wandb_mode(mode: str) -> str:
    if mode != "auto":
        return mode
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    try:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            return "online"
    except (FileNotFoundError, netrc.NetrcParseError):
        pass
    return "offline"


def wandb_kwargs(project: str, mode: str) -> dict[str, str]:
    kwargs = {"project": os.environ.get("WANDB_PROJECT", project), "mode": resolve_wandb_mode(mode)}
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        kwargs["entity"] = entity
    return kwargs
