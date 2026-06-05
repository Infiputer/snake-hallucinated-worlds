from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SnakeTransitionFile:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.frames = np.load(self.path / "frames.npy", mmap_mode="r")
        self.context_indices = np.load(self.path / "context_indices.npy", mmap_mode="r")
        self.next_frame_indices = np.load(self.path / "next_frame_indices.npy", mmap_mode="r")
        self.actions = np.load(self.path / "actions.npy", mmap_mode="r")
        self.rewards = np.load(self.path / "rewards.npy", mmap_mode="r")
        self.dones = np.load(self.path / "dones.npy", mmap_mode="r")
        self.lengths = np.load(self.path / "lengths.npy", mmap_mode="r")
        self.prev_rewards = np.load(self.path / "prev_rewards.npy", mmap_mode="r")
        self.episode_ids = np.load(self.path / "episode_ids.npy", mmap_mode="r")
        self.policy_ids = np.load(self.path / "policy_ids.npy", mmap_mode="r")

    def split_indices(self, split: str, train_fraction: float = 0.9) -> np.ndarray:
        episodes = np.unique(self.episode_ids)
        cutoff = int(len(episodes) * train_fraction)
        train_eps = set(int(x) for x in episodes[:cutoff])
        mask = np.array([int(ep) in train_eps for ep in self.episode_ids])
        if split == "train":
            return np.flatnonzero(mask)
        if split in ("val", "valid", "validation"):
            return np.flatnonzero(~mask)
        if split == "all":
            return np.arange(len(self.episode_ids))
        raise ValueError(f"unknown split: {split}")


class SnakeWorldModelDataset(Dataset):
    def __init__(self, path: str | Path, context: int, split: str = "train"):
        self.file = SnakeTransitionFile(path)
        self.context = int(context)
        if self.context not in (1, 2, 5):
            raise ValueError("context must be one of 1, 2, 5")
        self.indices = self.file.split_indices(split)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[i])
        context_idx = self.file.context_indices[idx, -self.context:]
        context = self.file.frames[context_idx].astype(np.float32) / 255.0
        next_frame = self.file.frames[int(self.file.next_frame_indices[idx])].astype(np.float32) / 255.0
        return {
            "context": torch.from_numpy(context).permute(0, 3, 1, 2),
            "action": torch.tensor(self.file.actions[idx], dtype=torch.long),
            "next_frame": torch.from_numpy(next_frame).permute(2, 0, 1),
            "reward": torch.tensor(self.file.rewards[idx], dtype=torch.float32),
            "done": torch.tensor(self.file.dones[idx], dtype=torch.float32),
            "length": torch.tensor(self.file.lengths[idx], dtype=torch.float32),
            "prev_reward": torch.tensor(self.file.prev_rewards[idx], dtype=torch.float32),
        }
