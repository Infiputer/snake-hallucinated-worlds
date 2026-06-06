from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import set_seed, utc_stamp, write_json
from .pacman_env import ACTION_VECTORS, DOWN, LEFT, RIGHT, UP, PacmanEnv

MAX_CONTEXT = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Pac-Man event world-model transitions")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--max-transitions", type=int, default=50000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", default=None)
    p.add_argument("--random-map", action="store_true")
    p.add_argument("--progress-every", type=int, default=5000)
    return p.parse_args()


def safe_actions(env: PacmanEnv) -> list[int]:
    out = []
    for action, (dx, dy) in enumerate(ACTION_VECTORS):
        nxt = (env.player[0] + int(dx), env.player[1] + int(dy))
        if not env._blocked(nxt):
            out.append(action)
    return out or [RIGHT]


def greedy_action(env: PacmanEnv, rng: np.random.Generator) -> int:
    safe = safe_actions(env)
    if not env.pellets:
        return int(rng.choice(safe))
    target = min(env.pellets, key=lambda p: abs(p[0] - env.player[0]) + abs(p[1] - env.player[1]))
    scored = []
    for action in safe:
        dx, dy = ACTION_VECTORS[action]
        nxt = (env.player[0] + int(dx), env.player[1] + int(dy))
        ghost_dist = min(abs(nxt[0] - g[0]) + abs(nxt[1] - g[1]) for g in env.ghosts)
        pellet_dist = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
        scored.append((pellet_dist - 0.8 * min(ghost_dist, 4), action))
    scored.sort()
    return int(rng.choice(safe) if rng.random() < 0.20 else scored[0][1])


def sample_action(policy: str, env: PacmanEnv, rng: np.random.Generator) -> int:
    if policy == "random":
        return int(rng.integers(0, 4))
    if policy == "safe_random":
        return int(rng.choice(safe_actions(env)))
    if policy == "greedy":
        return greedy_action(env, rng)
    raise ValueError(policy)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out or f"runs/datasets/pacman_{utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = ("random", "safe_random", "greedy")

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
        env = PacmanEnv(seed=args.seed + ep, random_map=args.random_map)
        result = env.reset()
        frames.append(result.frame.astype(np.uint8))
        history = [len(frames) - 1 for _ in range(MAX_CONTEXT)]
        prev_reward = 0.0
        policy = policies[ep % len(policies)]
        for _ in range(args.max_steps):
            action = sample_action(policy, env, rng)
            result = env.step(action)
            frames.append(result.frame.astype(np.uint8))
            next_idx = len(frames) - 1
            context_indices.append(np.asarray(history[-MAX_CONTEXT:], dtype=np.int64))
            next_frame_indices.append(next_idx)
            actions.append(int(action))
            rewards.append(np.float32(result.reward))
            dones.append(np.float32(result.done))
            lengths.append(np.float32(result.pellets))
            prev_rewards.append(np.float32(prev_reward))
            episode_ids.append(ep)
            policy_ids.append(policies.index(policy))
            statuses.append(result.status)
            history.append(next_idx)
            prev_reward = result.reward
            if args.progress_every and len(actions) % args.progress_every == 0:
                print(f"transitions={len(actions)} frames={len(frames)} episode={ep}", flush=True)
            if result.done or (args.max_transitions and len(actions) >= args.max_transitions):
                break

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
        "format": "indexed_npy_event_pacman_v1",
        "path": out_dir.as_posix(),
        "episodes_requested": args.episodes,
        "episodes_recorded": int(max(episode_ids) + 1 if episode_ids else 0),
        "transitions": len(actions),
        "frames": len(frames),
        "frame_size": 256,
        "max_context": MAX_CONTEXT,
        "random_map": bool(args.random_map),
        "policies": list(policies),
        "reward": "pellet event +1, death -1, otherwise 0",
        "status_counts": status_counts,
    })
    print(out_dir.as_posix())
    print(f"transitions={len(actions)} frames={len(frames)} status_counts={status_counts}")


if __name__ == "__main__":
    main()
