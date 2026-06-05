from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .common import RewardConfig, count_parameters, set_seed, shaped_reward, write_json
from .env import SnakeEnv
from .models import CNNPolicy
from .train_policy import HallucinatedBatchEnv, load_world_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a CNN policy in real or hallucinated Snake")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("real", "hallucinated"), default="real")
    p.add_argument("--world-model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--done-threshold", type=float, default=0.8)
    p.add_argument("--reward-mode", choices=("wm", "length_delta", "mixed"), default="wm")
    p.add_argument("--length-reward-scale", type=float, default=1.0)
    p.add_argument("--wm-reward-scale", type=float, default=1.0)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--length-delta-clamp", type=float, default=0.0)
    p.add_argument("--ppo-reward-clamp", type=float, default=5.0)
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
    reward_cfg = RewardConfig()
    rows = []
    for ep in range(cfg.episodes):
        env = SnakeEnv(seed=cfg.seed + ep)
        result = env.reset()
        total_reward = 0.0
        apples = 0
        for step in range(cfg.max_steps):
            action = int(choose_action(policy, result.frame, device, cfg.sample).item())
            result = env.step(action)
            reward = shaped_reward(result.reward, result.status, reward_cfg)
            total_reward += reward
            apples += 1 if result.reward > 0 else 0
            if result.done:
                break
        rows.append({"episode": ep, "return": total_reward, "steps": step + 1, "length": result.length, "apples": apples, "status": result.status, "win": int(result.status == "win"), "dead": int(result.status == "dead")})
    summary = {"mode": "real", "episodes": cfg.episodes, "mean_return": float(np.mean([r["return"] for r in rows])), "mean_steps": float(np.mean([r["steps"] for r in rows])), "mean_length": float(np.mean([r["length"] for r in rows])), "max_length": float(np.max([r["length"] for r in rows])), "mean_apples": float(np.mean([r["apples"] for r in rows])), "win_rate": float(np.mean([r["win"] for r in rows])), "death_rate": float(np.mean([r["dead"] for r in rows]))}
    return rows, summary


@torch.no_grad()
def eval_hallucinated(policy: CNNPolicy, cfg: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    if not cfg.world_model or not cfg.dataset:
        raise ValueError("hallucinated mode requires --world-model and --dataset")
    wm = load_world_model(cfg.world_model, device)
    env = HallucinatedBatchEnv(
        wm,
        cfg.dataset,
        1,
        cfg.max_steps,
        cfg.done_threshold,
        device,
        cfg.seed,
        cfg.reward_mode,
        cfg.length_reward_scale,
        cfg.wm_reward_scale,
        cfg.death_penalty,
        cfg.length_delta_clamp,
        cfg.ppo_reward_clamp,
    )
    rows = []
    for ep in range(cfg.episodes):
        env.reset(torch.tensor([0], device=device))
        total_reward = 0.0
        for step in range(cfg.max_steps):
            action = choose_action(policy, env.obs, device, cfg.sample)
            _, reward, done = env.step(action)
            total_reward += float(reward.item())
            if bool(done.item()):
                break
        rows.append({"episode": ep, "return": total_reward, "steps": step + 1, "done": int(bool(done.item()))})
    summary = {"mode": "hallucinated", "reward_mode": cfg.reward_mode, "episodes": cfg.episodes, "mean_return": float(np.mean([r["return"] for r in rows])), "mean_steps": float(np.mean([r["steps"] for r in rows])), "done_rate": float(np.mean([r["done"] for r in rows]))}
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
    summary.update({"policy": cfg.policy, "policy_variant": ckpt["policy_variant"], "policy_params": count_parameters(policy), "world_model": ckpt.get("world_model"), "csv": csv_path.as_posix()})
    write_json(out_dir / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
