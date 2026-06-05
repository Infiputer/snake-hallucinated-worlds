from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from .common import read_json, set_seed, wandb_kwargs, write_json
from .env import DOWN, LEFT, OPPOSITE, RIGHT, UP, SnakeEnv
from .train_policy import load_world_model


ACTIONS = (UP, DOWN, LEFT, RIGHT)
ACTION_VECTORS = {UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scalar-consistency diagnostic and PPO reward-objective ablation")
    p.add_argument("--root", default="runs/reward_length_ablation")
    p.add_argument("--dataset", required=True)
    p.add_argument("--world-model", required=True)
    p.add_argument("--policy", default="medium")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--eval-max-steps", type=int, default=256)
    p.add_argument("--diagnostic-episodes-per-mode", type=int, default=50)
    p.add_argument("--diagnostic-max-steps", type=int, default=96)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-v2")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def frame_tensor(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)


def greedy_toward_apple(env: SnakeEnv) -> int:
    hx, hy = env.snake[0]
    ax, ay = env.apples[0] if env.apples else (hx, hy)
    prefs: list[int] = []
    if ax > hx:
        prefs.append(RIGHT)
    if ax < hx:
        prefs.append(LEFT)
    if ay > hy:
        prefs.append(DOWN)
    if ay < hy:
        prefs.append(UP)
    prefs.extend(ACTIONS)
    for action in prefs:
        if action != OPPOSITE.get(env.direction):
            return int(action)
    return int(env.direction)


def safe_random(env: SnakeEnv, rng: np.random.Generator) -> int:
    choices = [a for a in ACTIONS if a != OPPOSITE.get(env.direction)]
    rng.shuffle(choices)
    hx, hy = env.snake[0]
    for action in choices:
        dx, dy = ACTION_VECTORS[action]
        nx, ny = hx + dx, hy + dy
        body = env.snake[:-1]
        if 0 <= nx < 16 and 0 <= ny < 16 and (nx, ny) not in env._rock_set and (nx, ny) not in body:
            return int(action)
    return int(choices[0])


def choose_action(mode: str, env: SnakeEnv, rng: np.random.Generator) -> int:
    if mode == "random":
        valid = [a for a in ACTIONS if a != OPPOSITE.get(env.direction)]
        return int(rng.choice(valid))
    if mode == "safe_random":
        return safe_random(env, rng)
    if mode == "greedy":
        return greedy_toward_apple(env)
    if mode == "mixed":
        return safe_random(env, rng) if rng.random() < 0.5 else greedy_toward_apple(env)
    raise ValueError(mode)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_scalar_diagnostic(args: argparse.Namespace, root: Path) -> dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_world_model(args.world_model, device)
    rng = np.random.default_rng(args.seed)
    modes = ("random", "safe_random", "greedy", "mixed")
    step_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    counterexamples: list[dict[str, object]] = []
    example_rollout: list[dict[str, float]] = []

    for mode in modes:
        for ep in range(args.diagnostic_episodes_per_mode):
            env = SnakeEnv(seed=args.seed + ep + 1000 * modes.index(mode))
            result = env.reset()
            context = torch.stack([frame_tensor(result.frame) for _ in range(model.context)], dim=0).unsqueeze(0).to(device)
            prev_reward = torch.tensor([0.0], device=device)
            cumulative_pred_reward = 0.0
            ep_length_errors: list[float] = []
            ep_reward_errors: list[float] = []
            apples = 0
            for step in range(1, args.diagnostic_max_steps + 1):
                action = choose_action(mode, env, rng)
                with torch.no_grad():
                    out = model(context, prev_reward, torch.tensor([action], device=device))
                    pred_reward = float(out["reward"].item())
                    pred_length = float(out["length"].item())
                    pred_done = float(torch.sigmoid(out["done_logit"]).item())
                    pred_frame = out["frame"].detach()
                result = env.step(action)
                cumulative_pred_reward += pred_reward
                reward_integrated_length = 3.0 + cumulative_pred_reward
                true_length = float(result.length)
                direct_error = abs(pred_length - true_length)
                reward_error = abs(reward_integrated_length - true_length)
                apples += int(result.reward > 0)
                row = {
                    "mode": mode,
                    "episode": ep,
                    "step": step,
                    "true_reward": float(result.reward),
                    "predicted_reward": pred_reward,
                    "true_length": true_length,
                    "predicted_length": pred_length,
                    "reward_integrated_length": reward_integrated_length,
                    "direct_length_abs_error": direct_error,
                    "reward_integrated_abs_error": reward_error,
                    "predicted_done_probability": pred_done,
                    "status": result.status,
                }
                step_rows.append(row)
                ep_length_errors.append(direct_error)
                ep_reward_errors.append(reward_error)
                if direct_error < reward_error and len(counterexamples) < 40:
                    counterexamples.append(row)
                if mode == "greedy" and ep == 0:
                    example_rollout.append(row)
                context = torch.cat([context[:, 1:], pred_frame.unsqueeze(1)], dim=1) if model.context > 1 else pred_frame.unsqueeze(1)
                prev_reward = torch.tensor([pred_reward], device=device)
                if result.done:
                    break
            episode_rows.append(
                {
                    "mode": mode,
                    "episode": ep,
                    "steps": len(ep_length_errors),
                    "apples": apples,
                    "direct_length_mae": float(np.mean(ep_length_errors)),
                    "reward_integrated_mae": float(np.mean(ep_reward_errors)),
                    "direct_length_final_abs_error": float(ep_length_errors[-1]),
                    "reward_integrated_final_abs_error": float(ep_reward_errors[-1]),
                }
            )

    write_csv(root / "scalar_consistency_steps.csv", step_rows)
    write_csv(root / "scalar_consistency_episodes.csv", episode_rows)
    write_csv(root / "scalar_consistency_counterexamples.csv", counterexamples)

    direct = np.asarray([float(r["direct_length_abs_error"]) for r in step_rows], dtype=np.float32)
    reward = np.asarray([float(r["reward_integrated_abs_error"]) for r in step_rows], dtype=np.float32)
    apple_direct = np.asarray([float(r["direct_length_abs_error"]) for r in step_rows if float(r["true_reward"]) > 0], dtype=np.float32)
    apple_reward = np.asarray([float(r["reward_integrated_abs_error"]) for r in step_rows if float(r["true_reward"]) > 0], dtype=np.float32)
    summary = {
        "episodes": float(len(episode_rows)),
        "steps": float(len(step_rows)),
        "direct_length_step_mae": float(direct.mean()),
        "reward_integrated_step_mae": float(reward.mean()),
        "direct_length_step_median_ae": float(np.median(direct)),
        "reward_integrated_step_median_ae": float(np.median(reward)),
        "direct_length_apple_step_mae": float(apple_direct.mean()) if apple_direct.size else float("nan"),
        "reward_integrated_apple_step_mae": float(apple_reward.mean()) if apple_reward.size else float("nan"),
        "reward_integrated_better_step_fraction": float(np.mean(reward < direct)),
        "direct_length_better_step_fraction": float(np.mean(direct < reward)),
        "reward_integrated_better_episode_count": float(sum(float(r["reward_integrated_mae"]) < float(r["direct_length_mae"]) for r in episode_rows)),
        "direct_length_better_episode_count": float(sum(float(r["direct_length_mae"]) < float(r["reward_integrated_mae"]) for r in episode_rows)),
    }
    write_json(root / "scalar_consistency_summary.json", summary)

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=180)
    labels = ["All steps", "Apple steps"]
    direct_vals = [summary["direct_length_step_mae"], summary["direct_length_apple_step_mae"]]
    reward_vals = [summary["reward_integrated_step_mae"], summary["reward_integrated_apple_step_mae"]]
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, direct_vals, width, label="Predicted length head", color="#3a6ea5")
    ax.bar(x + width / 2, reward_vals, width, label="3 + cumulative predicted reward", color="#d9822b")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean absolute error vs true length")
    ax.set_title("Scalar consistency diagnostic")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "scalar_consistency_mae.png", bbox_inches="tight")
    plt.close(fig)

    if example_rollout:
        fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=180)
        xs = [int(r["step"]) for r in example_rollout]
        ax.plot(xs, [float(r["true_length"]) for r in example_rollout], label="True length", color="black", linewidth=2)
        ax.plot(xs, [float(r["predicted_length"]) for r in example_rollout], label="Predicted length head", color="#3a6ea5")
        ax.plot(xs, [float(r["reward_integrated_length"]) for r in example_rollout], label="3 + cumulative predicted reward", color="#d9822b")
        ax.set_xlabel("Step")
        ax.set_ylabel("Snake length estimate")
        ax.set_title("Example scalar consistency rollout")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "scalar_consistency_rollout.png", bbox_inches="tight")
        plt.close(fig)

    run = wandb.init(name="scalar_consistency", config=vars(args), **wandb_kwargs(args.project, args.wandb_mode))
    wandb.log({f"scalar/{k}": v for k, v in summary.items()})
    for fig_name in ("scalar_consistency_mae.png", "scalar_consistency_rollout.png"):
        fig_path = fig_dir / fig_name
        if fig_path.exists():
            wandb.log({f"scalar/{fig_name}": wandb.Image(str(fig_path))})
    run.finish()
    return summary


def summary_value(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    return float(read_json(path).get(key, default))


def run_policy_ablation(args: argparse.Namespace, root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in ("wm", "length_delta", "mixed"):
        pol_dir = root / "policies" / f"{args.policy}_{mode}"
        pol_ckpt = pol_dir / "latest.pt"
        if not (args.skip_existing and pol_ckpt.exists()):
            cmd = [
                sys.executable,
                "-m",
                "snake_wm_v2.train_policy",
                "--dataset",
                args.dataset,
                "--world-model",
                args.world_model,
                "--out",
                str(pol_dir),
                "--policy",
                args.policy,
                "--updates",
                str(args.updates),
                "--num-envs",
                str(args.num_envs),
                "--rollout-steps",
                str(args.rollout_steps),
                "--minibatch-size",
                str(args.minibatch_size),
                "--reward-mode",
                mode,
                "--done-threshold",
                "0.8",
                "--wandb-mode",
                args.wandb_mode,
                "--project",
                args.project,
            ]
            run(cmd)
        hall_dir = root / "evals" / f"hall_{mode}"
        real_dir = root / "evals" / f"real_{mode}"
        if not (args.skip_existing and (hall_dir / "summary.json").exists()):
            run(
                [
                    sys.executable,
                    "-m",
                    "snake_wm_v2.evaluate",
                    "--mode",
                    "hallucinated",
                    "--dataset",
                    args.dataset,
                    "--world-model",
                    args.world_model,
                    "--policy",
                    str(pol_ckpt),
                    "--out",
                    str(hall_dir),
                    "--episodes",
                    str(args.eval_episodes),
                    "--max-steps",
                    str(args.eval_max_steps),
                    "--done-threshold",
                    "0.8",
                ]
            )
        if not (args.skip_existing and (real_dir / "summary.json").exists()):
            run(
                [
                    sys.executable,
                    "-m",
                    "snake_wm_v2.evaluate",
                    "--mode",
                    "real",
                    "--policy",
                    str(pol_ckpt),
                    "--out",
                    str(real_dir),
                    "--episodes",
                    str(args.eval_episodes),
                    "--max-steps",
                    str(args.eval_max_steps),
                ]
            )
        rows.append(
            {
                "reward_mode": mode,
                "policy": args.policy,
                "policy_dir": pol_dir.as_posix(),
                "hallucinated_reward_return": summary_value(hall_dir / "summary.json", "mean_return"),
                "hallucinated_done_rate": summary_value(hall_dir / "summary.json", "done_rate"),
                "real_return": summary_value(real_dir / "summary.json", "mean_return"),
                "real_apples": summary_value(real_dir / "summary.json", "mean_apples"),
                "real_death_rate": summary_value(real_dir / "summary.json", "death_rate"),
                "real_mean_steps": summary_value(real_dir / "summary.json", "mean_steps"),
            }
        )
        write_csv(root / "reward_objective_ablation.csv", rows)

    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["reward_mode"]) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    ax.bar(x - 0.18, [float(r["real_return"]) for r in rows], 0.36, label="Real return", color="#3a6ea5")
    ax.bar(x + 0.18, [float(r["real_apples"]) for r in rows], 0.36, label="Real apples", color="#d9822b")
    ax.set_xticks(x, labels)
    ax.set_ylabel("True-simulator evaluation")
    ax.set_title("PPO reward-objective ablation")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "reward_objective_ablation.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    scalar_summary = run_scalar_diagnostic(args, root)
    ablation_rows = run_policy_ablation(args, root)
    write_json(root / "summary.json", {"args": vars(args), "scalar_consistency": scalar_summary, "reward_objective_ablation": ablation_rows})
    print(read_json(root / "summary.json"))


if __name__ == "__main__":
    main()
