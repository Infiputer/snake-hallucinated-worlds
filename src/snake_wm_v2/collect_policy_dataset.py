from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import RewardConfig, set_seed, shaped_reward, write_json
from .env import SnakeEnv
from .evaluate import choose_action, load_policy
from .generate_dataset import MAX_CONTEXT, assert_no_terminal_tint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect no-overlay real-simulator Snake transitions from a trained CNN policy")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--max-transitions", type=int, default=1000)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--sample", action="store_true")
    p.add_argument("--death-penalty", type=float, default=-1.0)
    p.add_argument("--win-reward", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_policy(args.policy, device)
    reward_cfg = RewardConfig(death_penalty=args.death_penalty, win_reward=args.win_reward)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    context_indices: list[np.ndarray] = []
    next_frame_indices: list[int] = []
    actions: list[int] = []
    rewards: list[np.float32] = []
    dones: list[np.float32] = []
    lengths: list[np.float32] = []
    prev_rewards: list[np.float32] = []
    episode_ids: list[int] = []
    policy_ids: list[int] = []
    statuses: list[str] = []

    for ep in range(args.episodes):
        if args.max_transitions and len(actions) >= args.max_transitions:
            break
        env = SnakeEnv(seed=args.seed + ep)
        result = env.reset()
        frames.append(result.frame.astype(np.uint8))
        frame_history = [len(frames) - 1 for _ in range(MAX_CONTEXT)]
        prev_reward = 0.0
        for _ in range(args.max_steps):
            action = int(choose_action(policy, result.frame, device, args.sample).item())
            result = env.step(action)
            reward = shaped_reward(result.reward, result.status, reward_cfg)
            frames.append(result.frame.astype(np.uint8))
            next_idx = len(frames) - 1
            context_indices.append(np.asarray(frame_history[-MAX_CONTEXT:], dtype=np.int64))
            actions.append(action)
            next_frame_indices.append(next_idx)
            rewards.append(np.float32(reward))
            dones.append(np.float32(result.done))
            lengths.append(np.float32(result.length))
            prev_rewards.append(np.float32(prev_reward))
            episode_ids.append(ep)
            policy_ids.append(100)
            statuses.append(result.status)
            frame_history.append(next_idx)
            prev_reward = reward
            if result.done or (args.max_transitions and len(actions) >= args.max_transitions):
                break

    if not actions:
        raise RuntimeError("policy collection produced zero transitions")
    assert_no_terminal_tint(frames)
    np.save(out_dir / "frames.npy", np.stack(frames, axis=0))
    np.save(out_dir / "context_indices.npy", np.stack(context_indices, axis=0))
    np.save(out_dir / "next_frame_indices.npy", np.asarray(next_frame_indices, dtype=np.int64))
    np.save(out_dir / "actions.npy", np.asarray(actions, dtype=np.int64))
    np.save(out_dir / "rewards.npy", np.asarray(rewards, dtype=np.float32))
    np.save(out_dir / "dones.npy", np.asarray(dones, dtype=np.float32))
    np.save(out_dir / "lengths.npy", np.asarray(lengths, dtype=np.float32))
    np.save(out_dir / "prev_rewards.npy", np.asarray(prev_rewards, dtype=np.float32))
    np.save(out_dir / "episode_ids.npy", np.asarray(episode_ids, dtype=np.int64))
    np.save(out_dir / "policy_ids.npy", np.asarray(policy_ids, dtype=np.int64))
    status_counts = {s: statuses.count(s) for s in sorted(set(statuses))}
    write_json(out_dir / "dataset_meta.json", {
        "format": "indexed_npy_v2_policy_collection_no_terminal_overlay",
        "path": out_dir.as_posix(),
        "source_policy": args.policy,
        "source_policy_variant": ckpt.get("policy_variant"),
        "episodes_requested": args.episodes,
        "episodes_recorded": int(max(episode_ids) + 1 if episode_ids else 0),
        "transitions": len(actions),
        "frames": len(frames),
        "max_context": MAX_CONTEXT,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "sample": bool(args.sample),
        "reward": reward_cfg.__dict__,
        "status_counts": status_counts,
        "corner_colors": sorted([list(c) for c in set(tuple(map(int, f[0, 0])) for f in frames)]),
    })
    print(out_dir.as_posix())
    print(f"transitions={len(actions)} frames={len(frames)} status_counts={status_counts}")


if __name__ == "__main__":
    main()
