from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from .common import read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run focused no-overlay v2 sweep")
    p.add_argument("--config", default="configs/focused_v2.json")
    p.add_argument("--root", default="runs/focused_v2")
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def summary_value(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    return float(read_json(path).get(key, default))


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(cfg["dataset"]["out"])
    project = cfg.get("project", "snake-hallucinated-worlds-v2")
    if not (dataset / "dataset_meta.json").exists():
        d = cfg["dataset"]
        run([sys.executable, "-m", "snake_wm_v2.generate_dataset", "--out", str(dataset), "--episodes", str(d["episodes"]), "--max-steps", str(d["max_steps"]), "--max-transitions", str(d["max_transitions"]), "--seed", str(d["seed"])])
    else:
        print(f"dataset exists: {dataset}")
    rows: list[dict[str, str | float | int]] = []
    wm_cfg = cfg["world_models"]
    pol_cfg = cfg["policies"]
    ev_cfg = cfg["eval"]
    for wm_variant in wm_cfg["variants"]:
        for context in wm_cfg["contexts"]:
            wm_dir = root / "world_models" / f"{wm_variant}_ctx{context}"
            wm_ckpt = wm_dir / "latest.pt"
            if not (args.skip_existing and wm_ckpt.exists()):
                run([sys.executable, "-m", "snake_wm_v2.train_world_model", "--dataset", str(dataset), "--out", str(wm_dir), "--variant", wm_variant, "--context", str(context), "--steps", str(wm_cfg["steps"]), "--batch-size", str(wm_cfg["batch_size"]), "--wandb-mode", args.wandb_mode, "--project", project])
            for policy_variant in pol_cfg["variants"]:
                tag = f"{wm_variant}_ctx{context}_{policy_variant}"
                pol_dir = root / "policies" / tag
                pol_ckpt = pol_dir / "latest.pt"
                if not (args.skip_existing and pol_ckpt.exists()):
                    run([sys.executable, "-m", "snake_wm_v2.train_policy", "--dataset", str(dataset), "--world-model", str(wm_ckpt), "--out", str(pol_dir), "--policy", policy_variant, "--updates", str(pol_cfg["updates"]), "--num-envs", str(pol_cfg["num_envs"]), "--rollout-steps", str(pol_cfg["rollout_steps"]), "--minibatch-size", str(pol_cfg["minibatch_size"]), "--done-threshold", str(pol_cfg.get("done_threshold", 0.8)), "--wandb-mode", args.wandb_mode, "--project", project])
                hall_dir = root / "evals" / f"hall_{tag}"
                real_dir = root / "evals" / f"real_{tag}"
                if not (args.skip_existing and (hall_dir / "summary.json").exists()):
                    run([sys.executable, "-m", "snake_wm_v2.evaluate", "--mode", "hallucinated", "--dataset", str(dataset), "--world-model", str(wm_ckpt), "--policy", str(pol_ckpt), "--out", str(hall_dir), "--episodes", str(ev_cfg["episodes"]), "--max-steps", str(ev_cfg["max_steps"]), "--done-threshold", str(pol_cfg.get("done_threshold", 0.8))])
                if not (args.skip_existing and (real_dir / "summary.json").exists()):
                    run([sys.executable, "-m", "snake_wm_v2.evaluate", "--mode", "real", "--policy", str(pol_ckpt), "--out", str(real_dir), "--episodes", str(ev_cfg["episodes"]), "--max-steps", str(ev_cfg["max_steps"])])
                hall_return = summary_value(hall_dir / "summary.json", "mean_return")
                real_return = summary_value(real_dir / "summary.json", "mean_return")
                rows.append({"world_model": wm_variant, "context": context, "policy": policy_variant, "policy_dir": pol_dir.as_posix(), "world_model_dir": wm_dir.as_posix(), "hallucinated_return": hall_return, "real_return": real_return, "gap": hall_return - real_return, "real_apples": summary_value(real_dir / "summary.json", "mean_apples"), "real_death_rate": summary_value(real_dir / "summary.json", "death_rate")})
                out_csv = root / "eval_summary.csv"
                with out_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                write_json(root / "summary.json", {"rows": rows, "dataset": dataset.as_posix(), "config": cfg})
    run([sys.executable, "-m", "snake_wm_v2.make_figures", "--dataset", str(dataset), "--world-model", str(root / "world_models" / "wm_2m_ctx1" / "latest.pt"), "--eval-summary", str(root / "eval_summary.csv"), "--out", str(root / "figures")])


if __name__ == "__main__":
    main()
