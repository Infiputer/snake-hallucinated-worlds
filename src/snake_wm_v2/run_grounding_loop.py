from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from .common import read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run iterative real-simulator grounding for one selected WM/CNN setting")
    p.add_argument("--root", default="runs/focused_v2")
    p.add_argument("--dataset", default="runs/datasets/snake_v2_main")
    p.add_argument("--out", default="runs/focused_v2/grounding")
    p.add_argument("--eval-summary", default=None)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--collect-episodes", type=int, default=40)
    p.add_argument("--collect-max-transitions", type=int, default=1000)
    p.add_argument("--fine-tune-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--policy-updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--done-threshold", type=float, default=0.8)
    p.add_argument("--select", choices=("gap", "hallucinated_return", "real_return"), default="gap")
    p.add_argument("--world-model", default=None)
    p.add_argument("--policy", default=None)
    p.add_argument("--policy-variant", default=None)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-v2")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def choose_seed(args: argparse.Namespace) -> dict[str, str]:
    if args.world_model and args.policy and args.policy_variant:
        return {"world_model_dir": args.world_model, "policy_dir": args.policy, "policy": args.policy_variant, "world_model": "manual", "context": "manual", "gap": "nan", "hallucinated_return": "nan", "real_return": "nan"}
    summary = Path(args.eval_summary or Path(args.root) / "eval_summary.csv")
    rows = read_rows(summary)
    if not rows:
        raise RuntimeError(f"no rows in {summary}")
    return max(rows, key=lambda r: float(r[args.select]))


def metric(path: Path, key: str) -> float:
    return float(read_json(path).get(key, 0.0))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    seed = choose_seed(args)

    wm_dir = Path(seed["world_model_dir"])
    policy_dir = Path(seed["policy_dir"])
    policy_variant = seed["policy"]
    wm_meta = read_json(wm_dir / "metadata.json")
    wm_cfg = wm_meta["model_config"]
    context = int(wm_cfg["context"])
    variant = str(wm_cfg["variant"])
    prev_wm = wm_dir / "latest.pt"
    prev_policy = policy_dir / "latest.pt"
    collections: list[Path] = []
    rows: list[dict[str, str | float | int]] = []

    write_json(out_root / "selection.json", {"selected": seed, "world_model_metadata": wm_meta, "args": vars(args)})

    for iteration in range(1, args.iterations + 1):
        collect_dir = out_root / "collections" / f"iter_{iteration:02d}"
        merged_dir = out_root / "datasets" / f"iter_{iteration:02d}"
        wm_next_dir = out_root / "world_models" / f"iter_{iteration:02d}_{variant}_ctx{context}"
        pol_next_dir = out_root / "policies" / f"iter_{iteration:02d}_{policy_variant}"
        hall_dir = out_root / "evals" / f"hall_iter_{iteration:02d}_{policy_variant}"
        real_dir = out_root / "evals" / f"real_iter_{iteration:02d}_{policy_variant}"

        if not (collect_dir / "dataset_meta.json").exists():
            run([sys.executable, "-m", "snake_wm_v2.collect_policy_dataset", "--policy", str(prev_policy), "--out", str(collect_dir), "--episodes", str(args.collect_episodes), "--max-steps", str(args.max_steps), "--max-transitions", str(args.collect_max_transitions), "--seed", str(8000 + iteration)])
        collections.append(collect_dir)

        if not (merged_dir / "dataset_meta.json").exists():
            run([sys.executable, "-m", "snake_wm_v2.merge_datasets", "--out", str(merged_dir), str(args.dataset), *[p.as_posix() for p in collections]])

        if not (wm_next_dir / "latest.pt").exists():
            run([sys.executable, "-m", "snake_wm_v2.train_world_model", "--dataset", str(merged_dir), "--out", str(wm_next_dir), "--variant", variant, "--context", str(context), "--steps", str(args.fine_tune_steps), "--batch-size", str(args.batch_size), "--init-checkpoint", str(prev_wm), "--wandb-mode", args.wandb_mode, "--project", args.project])

        if not (pol_next_dir / "latest.pt").exists():
            run([sys.executable, "-m", "snake_wm_v2.train_policy", "--dataset", str(merged_dir), "--world-model", str(wm_next_dir / "latest.pt"), "--out", str(pol_next_dir), "--policy", policy_variant, "--updates", str(args.policy_updates), "--num-envs", str(args.num_envs), "--rollout-steps", str(args.rollout_steps), "--minibatch-size", str(args.minibatch_size), "--done-threshold", str(args.done_threshold), "--wandb-mode", args.wandb_mode, "--project", args.project])

        if not (hall_dir / "summary.json").exists():
            run([sys.executable, "-m", "snake_wm_v2.evaluate", "--mode", "hallucinated", "--dataset", str(merged_dir), "--world-model", str(wm_next_dir / "latest.pt"), "--policy", str(pol_next_dir / "latest.pt"), "--out", str(hall_dir), "--episodes", str(args.eval_episodes), "--max-steps", str(args.max_steps), "--done-threshold", str(args.done_threshold)])
        if not (real_dir / "summary.json").exists():
            run([sys.executable, "-m", "snake_wm_v2.evaluate", "--mode", "real", "--policy", str(pol_next_dir / "latest.pt"), "--out", str(real_dir), "--episodes", str(args.eval_episodes), "--max-steps", str(args.max_steps)])

        collection_meta = read_json(collect_dir / "dataset_meta.json")
        rows.append({
            "iteration": iteration,
            "seed_world_model": seed.get("world_model", "manual"),
            "seed_context": seed.get("context", "manual"),
            "policy": policy_variant,
            "collected_transitions": int(collection_meta["transitions"]),
            "total_policy_collected_transitions": sum(int(read_json(p / "dataset_meta.json")["transitions"]) for p in collections),
            "hallucinated_return": metric(hall_dir / "summary.json", "mean_return"),
            "real_return": metric(real_dir / "summary.json", "mean_return"),
            "gap": metric(hall_dir / "summary.json", "mean_return") - metric(real_dir / "summary.json", "mean_return"),
            "real_apples": metric(real_dir / "summary.json", "mean_apples"),
            "real_death_rate": metric(real_dir / "summary.json", "death_rate"),
            "world_model_dir": wm_next_dir.as_posix(),
            "policy_dir": pol_next_dir.as_posix(),
        })
        with (out_root / "grounding_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out_root / "summary.json", {"selection": seed, "rows": rows, "args": vars(args)})
        prev_wm = wm_next_dir / "latest.pt"
        prev_policy = pol_next_dir / "latest.pt"


if __name__ == "__main__":
    main()
