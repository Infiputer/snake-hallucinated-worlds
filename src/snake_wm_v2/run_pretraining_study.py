from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from .common import read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper 2 study: fixed world-model PPO pretraining before real-simulator PPO")
    p.add_argument("--root", default="runs/pretraining_study")
    p.add_argument("--dataset", required=True)
    p.add_argument("--world-model", required=True)
    p.add_argument("--policy", default="medium")
    p.add_argument("--wm-updates", type=int, default=250)
    p.add_argument("--real-updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-wm-pretraining-v1")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_curve(path: Path) -> list[dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def auc(curve: list[dict[str, float]], key: str = "eval/return_mean") -> float:
    if len(curve) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(curve, curve[1:]):
        width = b["env_steps"] - a["env_steps"]
        total += width * 0.5 * (a[key] + b[key])
    return float(total)


def make_outputs(root: Path, scratch_dir: Path, pretrained_dir: Path, wm_pretrain_dir: Path) -> None:
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    scratch = load_curve(scratch_dir / "eval_curve.csv")
    pretrained = load_curve(pretrained_dir / "eval_curve.csv")
    rows = []
    for name, curve, run_dir in [("real_from_scratch", scratch, scratch_dir), ("wm_pretrain_then_real", pretrained, pretrained_dir)]:
        last = curve[-1]
        rows.append(
            {
                "condition": name,
                "final_return": last["eval/return_mean"],
                "final_apples": last["eval/apples_mean"],
                "final_death_rate": last["eval/death_rate"],
                "return_auc": auc(curve),
                "env_steps": last["env_steps"],
                "run_dir": run_dir.as_posix(),
            }
        )
    csv_path = root / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "summary.json", {"rows": rows, "wm_pretrain_dir": wm_pretrain_dir.as_posix()})
    plt.figure(figsize=(7.0, 4.2))
    plt.plot([r["env_steps"] for r in scratch], [r["eval/return_mean"] for r in scratch], marker="o", label="Real PPO from scratch")
    plt.plot([r["env_steps"] for r in pretrained], [r["eval/return_mean"] for r in pretrained], marker="o", label="WM-pretrained, then real PPO")
    plt.xlabel("Real simulator environment steps")
    plt.ylabel("Evaluation return in true simulator")
    plt.title("Does fixed world-model pretraining improve real PPO?")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "real_sample_efficiency.png", dpi=180)
    plt.close()
    plt.figure(figsize=(5.8, 4.2))
    plt.bar([r["condition"] for r in rows], [r["return_auc"] for r in rows], color=["#476f95", "#d8872f"])
    plt.ylabel("Area under real-return learning curve")
    plt.title("Real-simulator sample efficiency")
    plt.xticks(rotation=10, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "return_auc.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    wm_pretrain_dir = root / "wm_pretrain"
    scratch_dir = root / "real_from_scratch"
    pretrained_dir = root / "wm_pretrain_then_real"
    if not (args.skip_existing and (wm_pretrain_dir / "latest.pt").exists()):
        run(
            [
                sys.executable,
                "-m",
                "snake_wm_v2.train_policy",
                "--dataset",
                args.dataset,
                "--world-model",
                args.world_model,
                "--out",
                str(wm_pretrain_dir),
                "--policy",
                args.policy,
                "--updates",
                str(args.wm_updates),
                "--num-envs",
                str(args.num_envs),
                "--rollout-steps",
                str(args.rollout_steps),
                "--minibatch-size",
                str(args.minibatch_size),
                "--done-threshold",
                "0.8",
                "--wandb-mode",
                args.wandb_mode,
                "--project",
                args.project,
            ]
        )
    for out_dir, init_policy in [(scratch_dir, None), (pretrained_dir, wm_pretrain_dir / "latest.pt")]:
        if args.skip_existing and (out_dir / "eval_curve.csv").exists():
            continue
        cmd = [
            sys.executable,
            "-m",
            "snake_wm_v2.train_real_policy",
            "--out",
            str(out_dir),
            "--policy",
            args.policy,
            "--updates",
            str(args.real_updates),
            "--num-envs",
            str(args.num_envs),
            "--rollout-steps",
            str(args.rollout_steps),
            "--minibatch-size",
            str(args.minibatch_size),
            "--eval-every",
            str(args.eval_every),
            "--eval-episodes",
            str(args.eval_episodes),
            "--seed",
            str(args.seed),
            "--wandb-mode",
            args.wandb_mode,
            "--project",
            args.project,
        ]
        if init_policy is not None:
            cmd.extend(["--init-policy", str(init_policy)])
        run(cmd)
    make_outputs(root, scratch_dir, pretrained_dir, wm_pretrain_dir)
    write_json(root / "metadata.json", {"args": vars(args), "world_model_meta": read_json(Path(args.world_model).parent / "metadata.json") if (Path(args.world_model).parent / "metadata.json").exists() else None})


if __name__ == "__main__":
    main()
