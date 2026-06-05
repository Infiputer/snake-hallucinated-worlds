from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import read_json, write_json


OBJECTIVE_LABELS = {
    "wm": "Predicted reward",
    "length_delta": "Unclamped length delta",
}
POLICY_ORDER = {"small": 0, "medium": 1, "large": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run matched PPO objective sweep for reward and unclamped length-head objectives")
    p.add_argument("--config", default="configs/focused_v2.json")
    p.add_argument("--root", default="runs/matched_objective_sweep")
    p.add_argument("--dataset", default=None)
    p.add_argument("--objectives", default="wm,length_delta")
    p.add_argument("--policy-updates", type=int, default=300)
    p.add_argument("--world-model-steps", type=int, default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--max-rows", type=int, default=1000000)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default=None)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def summary_value(path: Path, key: str, default: float = 0.0) -> float:
    if not path.exists():
        return default
    return float(read_json(path).get(key, default))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def objective_args(objective: str) -> list[str]:
    if objective == "length_delta":
        return ["--reward-mode", "length_delta", "--length-delta-clamp", "0", "--ppo-reward-clamp", "0"]
    if objective == "wm":
        return ["--reward-mode", "wm", "--ppo-reward-clamp", "5"]
    raise ValueError(objective)


def row_key(row: dict[str, object]) -> tuple[int, int, int, int]:
    wm = str(row["world_model"])
    wm_order = 0 if wm == "wm_1m" else 1
    return (wm_order, int(row["context"]), POLICY_ORDER.get(str(row["policy"]), 99), 0)


def make_scaling_figures(rows: list[dict[str, object]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    objectives = [o for o in ("wm", "length_delta") if any(r["objective"] == o for r in rows)]
    policies = [p for p in ("small", "medium", "large") if any(r["policy"] == p for r in rows)]
    contexts = sorted({int(r["context"]) for r in rows})
    models = [m for m in ("wm_1m", "wm_2m") if any(r["world_model"] == m for r in rows)]

    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(5.0 * max(len(models), 1), 3.9), dpi=180, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    for ax, model in zip(axes, models):
        for objective in objectives:
            vals = []
            for context in contexts:
                selected = [float(r["real_apples"]) for r in rows if r["world_model"] == model and r["objective"] == objective and int(r["context"]) == context]
                vals.append(float(np.mean(selected)) if selected else np.nan)
            ax.plot(contexts, vals, marker="o", linewidth=2, label=OBJECTIVE_LABELS.get(objective, objective))
        ax.set_title(model.replace("_", " "))
        ax.set_xlabel("Context frames")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean real apples across CNN sizes")
    axes[-1].legend(frameon=False)
    fig.suptitle("Objective scaling by world-model context")
    fig.tight_layout()
    fig.savefig(out / "matched_objective_context_scaling.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=180)
    x = np.arange(len(policies))
    width = 0.34
    for i, objective in enumerate(objectives):
        vals = []
        for policy in policies:
            selected = [float(r["real_apples"]) for r in rows if r["objective"] == objective and r["policy"] == policy]
            vals.append(float(np.mean(selected)) if selected else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=OBJECTIVE_LABELS.get(objective, objective))
    ax.set_xticks(x, policies)
    ax.set_ylabel("Mean real apples across WMs/contexts")
    ax.set_title("Objective scaling by CNN size")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "matched_objective_policy_scaling.png", bbox_inches="tight")
    plt.close(fig)

    reward = {(r["world_model"], int(r["context"]), r["policy"]): float(r["real_apples"]) for r in rows if r["objective"] == "wm"}
    length = {(r["world_model"], int(r["context"]), r["policy"]): float(r["real_apples"]) for r in rows if r["objective"] == "length_delta"}
    keys = [k for k in sorted(reward, key=lambda k: (k[0], k[1], POLICY_ORDER.get(str(k[2]), 99))) if k in length]
    if keys:
        y_labels = [f"{k[0].replace('_', ' ')} ctx{k[1]}" for k in sorted({(k[0], k[1]) for k in keys})]
        y_index = {label: i for i, label in enumerate(y_labels)}
        matrix = np.full((len(y_labels), len(policies)), np.nan, dtype=np.float32)
        for model, context, policy in keys:
            label = f"{str(model).replace('_', ' ')} ctx{context}"
            matrix[y_index[label], policies.index(str(policy))] = reward[(model, context, policy)] - length[(model, context, policy)]
        fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=180)
        im = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
        ax.set_xticks(np.arange(len(policies)), policies)
        ax.set_yticks(np.arange(len(y_labels)), y_labels)
        ax.set_title("Real-apples advantage: reward objective minus length objective")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, label="Delta apples")
        fig.tight_layout()
        fig.savefig(out / "matched_objective_delta_heatmap.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.7, 4.4), dpi=180)
        xs = [reward[k] for k in keys]
        ys = [length[k] for k in keys]
        ax.scatter(xs, ys, s=42, alpha=0.8, color="#3a6ea5")
        lim = max(xs + ys + [1.0])
        ax.plot([0, lim], [0, lim], color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("Real apples: predicted reward PPO")
        ax.set_ylabel("Real apples: unclamped length PPO")
        ax.set_title("Matched-cell objective comparison")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "matched_objective_scatter.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = read_json(args.config)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = Path(args.dataset or cfg["dataset"]["out"])
    project = args.project or cfg.get("project", "snake-hallucinated-worlds-v2")
    objectives = [x.strip() for x in args.objectives.split(",") if x.strip()]

    if not (dataset / "dataset_meta.json").exists():
        d = cfg["dataset"]
        run([sys.executable, "-m", "snake_wm_v2.generate_dataset", "--out", str(dataset), "--episodes", str(d["episodes"]), "--max-steps", str(d["max_steps"]), "--max-transitions", str(d["max_transitions"]), "--seed", str(d["seed"])])

    wm_cfg = cfg["world_models"]
    pol_cfg = cfg["policies"]
    ev_cfg = cfg["eval"]
    wm_steps = int(args.world_model_steps or wm_cfg["steps"])
    policy_updates = int(args.policy_updates)
    eval_episodes = int(args.eval_episodes or ev_cfg["episodes"])
    rows: list[dict[str, object]] = []

    for wm_variant in wm_cfg["variants"]:
        for context in wm_cfg["contexts"]:
            wm_dir = root / "world_models" / f"{wm_variant}_ctx{context}"
            wm_ckpt = wm_dir / "latest.pt"
            if not (args.skip_existing and wm_ckpt.exists()):
                run([sys.executable, "-m", "snake_wm_v2.train_world_model", "--dataset", str(dataset), "--out", str(wm_dir), "--variant", wm_variant, "--context", str(context), "--steps", str(wm_steps), "--batch-size", str(wm_cfg["batch_size"]), "--wandb-mode", args.wandb_mode, "--project", project])
            for objective in objectives:
                for policy_variant in pol_cfg["variants"]:
                    tag = f"{objective}_{wm_variant}_ctx{context}_{policy_variant}"
                    pol_dir = root / "policies" / tag
                    pol_ckpt = pol_dir / "latest.pt"
                    if not (args.skip_existing and pol_ckpt.exists()):
                        run(
                            [
                                sys.executable,
                                "-m",
                                "snake_wm_v2.train_policy",
                                "--dataset",
                                str(dataset),
                                "--world-model",
                                str(wm_ckpt),
                                "--out",
                                str(pol_dir),
                                "--policy",
                                policy_variant,
                                "--updates",
                                str(policy_updates),
                                "--num-envs",
                                str(pol_cfg["num_envs"]),
                                "--rollout-steps",
                                str(pol_cfg["rollout_steps"]),
                                "--minibatch-size",
                                str(pol_cfg["minibatch_size"]),
                                "--done-threshold",
                                str(pol_cfg.get("done_threshold", 0.8)),
                                "--wandb-mode",
                                args.wandb_mode,
                                "--project",
                                project,
                                *objective_args(objective),
                            ]
                        )
                    hall_dir = root / "evals" / f"hall_{tag}"
                    real_dir = root / "evals" / f"real_{tag}"
                    if not (args.skip_existing and (hall_dir / "summary.json").exists()):
                        run(
                            [
                                sys.executable,
                                "-m",
                                "snake_wm_v2.evaluate",
                                "--mode",
                                "hallucinated",
                                "--dataset",
                                str(dataset),
                                "--world-model",
                                str(wm_ckpt),
                                "--policy",
                                str(pol_ckpt),
                                "--out",
                                str(hall_dir),
                                "--episodes",
                                str(eval_episodes),
                                "--max-steps",
                                str(ev_cfg["max_steps"]),
                                "--done-threshold",
                                str(pol_cfg.get("done_threshold", 0.8)),
                                *objective_args(objective),
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
                                str(eval_episodes),
                                "--max-steps",
                                str(ev_cfg["max_steps"]),
                            ]
                        )
                    hall_return = summary_value(hall_dir / "summary.json", "mean_return")
                    real_return = summary_value(real_dir / "summary.json", "mean_return")
                    rows.append(
                        {
                            "objective": objective,
                            "objective_label": OBJECTIVE_LABELS.get(objective, objective),
                            "world_model": wm_variant,
                            "context": context,
                            "policy": policy_variant,
                            "policy_dir": pol_dir.as_posix(),
                            "world_model_dir": wm_dir.as_posix(),
                            "hallucinated_return": hall_return,
                            "real_return": real_return,
                            "gap": hall_return - real_return,
                            "real_apples": summary_value(real_dir / "summary.json", "mean_apples"),
                            "real_death_rate": summary_value(real_dir / "summary.json", "death_rate"),
                        }
                    )
                    rows = sorted(rows, key=lambda r: (str(r["objective"]), *row_key(r)))
                    write_rows(root / "objective_eval_summary.csv", rows)
                    write_json(root / "summary.json", {"rows": rows, "dataset": dataset.as_posix(), "config": cfg, "args": vars(args)})
                    make_scaling_figures(rows, root / "figures")

    make_scaling_figures(rows, root / "figures")


if __name__ == "__main__":
    main()
