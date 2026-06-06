from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .common import count_parameters, set_seed, write_json
from .models import CNNPolicy
from .pacman_env import PacmanEnv
from .train_event_policy import EventHallucinatedBatchEnv, load_event_world_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CNN policy trained in a Pac-Man event world model")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("real", "hallucinated"), default="real")
    p.add_argument("--world-model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--reward-decoder", choices=("hard", "prob"), default="hard")
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.add_argument("--seed", type=int, default=999)
    return p.parse_args()


def load_policy(path: str | Path, device: torch.device) -> tuple[CNNPolicy, dict]:
    ckpt = torch.load(path, map_location=device)
    policy = CNNPolicy(ckpt["policy_variant"]).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, ckpt


def frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def choose_action(policy: CNNPolicy, frame: np.ndarray | torch.Tensor, device: torch.device, sample: bool) -> int | torch.Tensor:
    obs = frame if isinstance(frame, torch.Tensor) else frame_tensor(frame, device)
    logits, _ = policy(obs)
    if sample:
        return Categorical(logits=logits).sample()
    return torch.argmax(logits, dim=1)


@torch.no_grad()
def eval_real(policy: CNNPolicy, cfg: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    rows = []
    for ep in range(cfg.episodes):
        env = PacmanEnv(seed=cfg.seed + ep, random_map=cfg.random_map)
        result = env.reset()
        total_reward = 0.0
        pellets = 0
        for step in range(cfg.max_steps):
            action = int(choose_action(policy, result.frame, device, cfg.sample).item())
            result = env.step(action)
            pellet = 1 if result.reward > 0 else 0
            dead = 1 if result.status == "dead" else 0
            reward = -1.0 if dead else float(pellet)
            total_reward += reward
            pellets += pellet
            if result.done:
                break
        rows.append({
            "episode": ep,
            "return": total_reward,
            "steps": step + 1,
            "pellets": pellets,
            "remaining_pellets": result.pellets,
            "status": result.status,
            "win": int(result.status == "win"),
            "dead": int(result.status == "dead"),
        })
    summary = {
        "mode": "real",
        "reward": "pellet event +1, death -1, no shaping",
        "random_map": bool(cfg.random_map),
        "episodes": cfg.episodes,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_pellets": float(np.mean([r["pellets"] for r in rows])),
        "mean_remaining_pellets": float(np.mean([r["remaining_pellets"] for r in rows])),
        "win_rate": float(np.mean([r["win"] for r in rows])),
        "death_rate": float(np.mean([r["dead"] for r in rows])),
    }
    return rows, summary


@torch.no_grad()
def eval_hallucinated(policy: CNNPolicy, cfg: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    if not cfg.world_model or not cfg.dataset:
        raise ValueError("hallucinated mode requires --world-model and --dataset")
    wm = load_event_world_model(cfg.world_model, device)
    env = EventHallucinatedBatchEnv(wm, cfg.dataset, 1, cfg.max_steps, device, cfg.seed, cfg.reward_decoder)
    rows = []
    for ep in range(cfg.episodes):
        env.reset(torch.tensor([0], device=device))
        total_reward = 0.0
        pellets = 0
        pellet_objective = 0.0
        dead = 0
        for step in range(cfg.max_steps):
            action = choose_action(policy, env.obs, device, cfg.sample)
            _, reward, done, info = env.step(action)
            total_reward += float(reward.item())
            pellets += int(info["apple"].item())
            pellet_objective += float(info["apple_reward"].item())
            dead = int(info["death"].item())
            if bool(done.item()):
                break
        rows.append({"episode": ep, "return": total_reward, "steps": step + 1, "predicted_pellets": pellets, "predicted_pellet_objective": pellet_objective, "predicted_dead": dead})
    summary = {
        "mode": "hallucinated",
        "reward": f"{cfg.reward_decoder} pellet-event reward, hard death class",
        "episodes": cfg.episodes,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_predicted_pellets": float(np.mean([r["predicted_pellets"] for r in rows])),
        "mean_predicted_pellet_objective": float(np.mean([r["predicted_pellet_objective"] for r in rows])),
        "death_rate": float(np.mean([r["predicted_dead"] for r in rows])),
    }
    return rows, summary


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_policy(cfg.policy, device)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = eval_real(policy, cfg, device) if cfg.mode == "real" else eval_hallucinated(policy, cfg, device)
    csv_path = out_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary.update({
        "policy": cfg.policy,
        "policy_variant": ckpt["policy_variant"],
        "policy_params": count_parameters(policy),
        "world_model": ckpt.get("world_model"),
        "csv": csv_path.as_posix(),
    })
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
