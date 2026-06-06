from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .common import count_parameters, set_seed, write_json
from .env import ACTION_VECTORS, BOARD, DOWN, INITIAL_APPLES, LEFT, OPPOSITE, RIGHT, ROCKS_SET, UP, SnakeEnv
from .models import CNNPolicy
from .train_event_policy import EventHallucinatedBatchEnv, load_event_world_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate event-policy PPO in real or hallucinated Snake")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("real", "hallucinated"), default="real")
    p.add_argument("--world-model", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--reward-decoder", choices=("hard", "prob"), default="hard")
    p.add_argument("--layout", choices=("fixed", "random_apples", "random_start_apples"), default="fixed")
    p.add_argument("--apple-count", type=int, default=len(INITIAL_APPLES))
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


def _snake_for_head(head: tuple[int, int], direction: int) -> list[tuple[int, int]]:
    dx, dy = ACTION_VECTORS[direction]
    hx, hy = head
    return [(hx, hy), (hx - int(dx), hy - int(dy)), (hx - 2 * int(dx), hy - 2 * int(dy))]


def _valid_cells(exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(x, y) for y in range(BOARD) for x in range(BOARD) if (x, y) not in exclude]


def randomize_layout(env: SnakeEnv, rng: np.random.Generator, layout: str, apple_count: int) -> None:
    if layout == "fixed":
        return
    if layout == "random_start_apples":
        candidates: list[tuple[list[tuple[int, int]], int]] = []
        for direction in (UP, DOWN, LEFT, RIGHT):
            for y in range(BOARD):
                for x in range(BOARD):
                    snake = _snake_for_head((x, y), direction)
                    if all(0 <= sx < BOARD and 0 <= sy < BOARD and (sx, sy) not in ROCKS_SET for sx, sy in snake):
                        candidates.append((snake, direction))
        snake, direction = candidates[int(rng.integers(0, len(candidates)))]
        env.snake = list(snake)
        env.direction = int(direction)
        env.last_action = int(direction)
    excluded = set(env.snake) | ROCKS_SET
    cells = _valid_cells(excluded)
    count = min(int(apple_count), len(cells))
    selected = rng.choice(len(cells), size=count, replace=False)
    env.apples = [cells[int(i)] for i in selected]
    env.status = "play"


@torch.no_grad()
def eval_real(policy: CNNPolicy, cfg: argparse.Namespace, device: torch.device) -> tuple[list[dict], dict]:
    rows = []
    for ep in range(cfg.episodes):
        env = SnakeEnv(seed=cfg.seed + ep)
        result = env.reset()
        rng = np.random.default_rng(cfg.seed + ep)
        randomize_layout(env, rng, cfg.layout, cfg.apple_count)
        result = env._result(0.0, False)
        total_reward = 0.0
        apples = 0
        for step in range(cfg.max_steps):
            action = int(choose_action(policy, result.frame, device, cfg.sample).item())
            result = env.step(action)
            apple = 1 if result.reward > 0 else 0
            dead = 1 if result.status == "dead" else 0
            reward = -1.0 if dead else float(apple)
            total_reward += reward
            apples += apple
            if result.done:
                break
        rows.append({
            "episode": ep,
            "layout": cfg.layout,
            "return": total_reward,
            "steps": step + 1,
            "apples": apples,
            "length": result.length,
            "status": result.status,
            "win": int(result.status == "win"),
            "dead": int(result.status == "dead"),
        })
    summary = {
        "mode": "real",
        "reward": "apple event +1, death -1, no win bonus",
        "layout": cfg.layout,
        "apple_count": int(cfg.apple_count),
        "episodes": cfg.episodes,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_apples": float(np.mean([r["apples"] for r in rows])),
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "max_length": float(np.max([r["length"] for r in rows])),
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
        apples = 0
        apple_objective = 0.0
        dead = 0
        for step in range(cfg.max_steps):
            action = choose_action(policy, env.obs, device, cfg.sample)
            _, reward, done, info = env.step(action)
            total_reward += float(reward.item())
            apples += int(info["apple"].item())
            apple_objective += float(info["apple_reward"].item())
            dead = int(info["death"].item())
            if bool(done.item()):
                break
        rows.append({"episode": ep, "return": total_reward, "steps": step + 1, "predicted_apples": apples, "predicted_apple_objective": apple_objective, "predicted_dead": dead})
    summary = {
        "mode": "hallucinated",
        "reward": f"{cfg.reward_decoder} apple-event reward, hard death class",
        "episodes": cfg.episodes,
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "mean_steps": float(np.mean([r["steps"] for r in rows])),
        "mean_predicted_apples": float(np.mean([r["predicted_apples"] for r in rows])),
        "mean_predicted_apple_objective": float(np.mean([r["predicted_apple_objective"] for r in rows])),
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
