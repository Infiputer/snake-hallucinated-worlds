from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import RewardConfig, set_seed, shaped_reward, utc_stamp, write_json
from .env import ACTION_VECTORS, BOARD, DOWN, INITIAL_APPLES, LEFT, RIGHT, ROCKS_SET, UP, OPPOSITE, SnakeEnv

ACTION_COUNT = 4
MAX_CONTEXT = 5
NORMAL_BG = {(170, 215, 81), (162, 209, 73)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate no-overlay Snake transitions")
    p.add_argument("--episodes", type=int, default=2500)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--max-transitions", type=int, default=50000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--death-penalty", type=float, default=-1.0)
    p.add_argument("--win-reward", type=float, default=5.0)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--randomize-apples", action="store_true", help="sample apple positions independently per episode")
    p.add_argument("--randomize-rocks", action="store_true", help="sample rock positions independently per episode; requires randomized apples")
    p.add_argument("--apple-count", type=int, default=len(INITIAL_APPLES))
    p.add_argument("--rock-count", type=int, default=len(ROCKS_SET))
    return p.parse_args()


def sample_cells(rng: np.random.Generator, count: int, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    cells = [(x, y) for y in range(BOARD) for x in range(BOARD) if (x, y) not in blocked]
    if count > len(cells):
        raise ValueError(f"cannot sample {count} cells with {len(blocked)} blocked cells on {BOARD}x{BOARD} board")
    idx = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in idx]


def apply_layout_randomization(env: SnakeEnv, rng: np.random.Generator, args: argparse.Namespace) -> None:
    snake_cells = set(env.snake)
    if args.randomize_rocks:
        env._rock_set = set(sample_cells(rng, args.rock_count, snake_cells))
    if args.randomize_apples:
        blocked = set(env.snake) | set(env._rock_set)
        env.apples = sample_cells(rng, args.apple_count, blocked)


def safe_actions(env: SnakeEnv) -> list[int]:
    actions = []
    for action in (UP, DOWN, LEFT, RIGHT):
        effective = env.direction if action == OPPOSITE.get(env.direction, env.direction) else action
        dx, dy = ACTION_VECTORS[effective]
        hx, hy = env.snake[0]
        nx, ny = hx + int(dx), hy + int(dy)
        body = env.snake[:-1]
        hit = nx < 0 or ny < 0 or nx >= 16 or ny >= 16 or (nx, ny) in env._rock_set or any(x == nx and y == ny for x, y in body)
        if not hit:
            actions.append(action)
    return actions or [env.last_action]


def greedy_action(env: SnakeEnv, rng: np.random.Generator) -> int:
    safe = safe_actions(env)
    if not env.apples:
        return int(rng.choice(safe))
    hx, hy = env.snake[0]
    apple = min(env.apples, key=lambda a: abs(a[0] - hx) + abs(a[1] - hy))
    ranked = []
    for action in safe:
        effective = env.direction if action == OPPOSITE.get(env.direction, env.direction) else action
        dx, dy = ACTION_VECTORS[effective]
        nx, ny = hx + int(dx), hy + int(dy)
        ranked.append((abs(apple[0] - nx) + abs(apple[1] - ny), action))
    ranked.sort(key=lambda x: x[0])
    return int(rng.choice(safe) if rng.random() < 0.15 else ranked[0][1])


def sample_action(policy: str, env: SnakeEnv, rng: np.random.Generator) -> int:
    if policy == "random":
        return int(rng.integers(0, ACTION_COUNT))
    if policy == "safe_random":
        return int(rng.choice(safe_actions(env)))
    if policy == "straight":
        return int(env.last_action if rng.random() < 0.8 else rng.integers(0, ACTION_COUNT))
    if policy == "greedy":
        return greedy_action(env, rng)
    raise ValueError(policy)


def assert_no_terminal_tint(frames: list[np.ndarray]) -> None:
    corners = [tuple(map(int, f[0, 0])) for f in frames]
    bad = sorted(set(c for c in corners if c not in NORMAL_BG))
    if bad:
        raise RuntimeError(f"unexpected global background/tint colors at corner pixel: {bad[:8]}")


def main() -> None:
    args = parse_args()
    if args.randomize_rocks and not args.randomize_apples:
        raise ValueError("--randomize-rocks requires --randomize-apples so apples cannot overlap randomized rocks")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out or f"runs/datasets/snake_v2_{utc_stamp()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    reward_cfg = RewardConfig(death_penalty=args.death_penalty, win_reward=args.win_reward)
    policies = ("random", "safe_random", "straight", "greedy")
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
        apply_layout_randomization(env, rng, args)
        result = env._result(0.0, False)
        frames.append(result.frame.astype(np.uint8))
        frame_history = [len(frames) - 1 for _ in range(MAX_CONTEXT)]
        prev_reward = 0.0
        policy = policies[ep % len(policies)]
        for _ in range(args.max_steps):
            action = sample_action(policy, env, rng)
            result = env.step(action)
            reward = shaped_reward(result.reward, result.status, reward_cfg)
            frames.append(result.frame.astype(np.uint8))
            next_idx = len(frames) - 1
            context_indices.append(np.asarray(frame_history[-MAX_CONTEXT:], dtype=np.int64))
            actions.append(int(action))
            next_frame_indices.append(next_idx)
            rewards.append(np.float32(reward))
            dones.append(np.float32(result.done))
            lengths.append(np.float32(result.length))
            prev_rewards.append(np.float32(prev_reward))
            episode_ids.append(ep)
            policy_ids.append(policies.index(policy))
            statuses.append(result.status)
            frame_history.append(next_idx)
            prev_reward = reward
            if args.progress_every and len(actions) % args.progress_every == 0:
                print(f"transitions={len(actions)} frames={len(frames)} episode={ep}", flush=True)
            if result.done or (args.max_transitions and len(actions) >= args.max_transitions):
                break

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
    write_json(out_dir / "dataset_meta.json", {"format": "indexed_npy_v2_no_terminal_overlay", "path": out_dir.as_posix(), "episodes_requested": args.episodes, "episodes_recorded": int(max(episode_ids) + 1 if episode_ids else 0), "transitions": len(actions), "frames": len(frames), "max_context": MAX_CONTEXT, "max_steps": args.max_steps, "seed": args.seed, "policies": list(policies), "reward": reward_cfg.__dict__, "layout_randomization": {"randomize_apples": bool(args.randomize_apples), "randomize_rocks": bool(args.randomize_rocks), "apple_count": int(args.apple_count), "rock_count": int(args.rock_count), "snake_start": "fixed"}, "status_counts": status_counts, "corner_colors": sorted([list(c) for c in set(tuple(map(int, f[0, 0])) for f in frames)])})
    print(out_dir.as_posix())
    print(f"transitions={len(actions)} frames={len(frames)} status_counts={status_counts}")


if __name__ == "__main__":
    main()
