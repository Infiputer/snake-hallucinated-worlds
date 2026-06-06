from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export final argmax Snake/Pac-Man paper tables and plots")
    p.add_argument("--paper", default="papers")
    p.add_argument("--snake-random-summary", default="../../snake_wm_v2/runs/local_event_random_eval_small_only_20260605_220114/small_hard_vs_prob_randomized_summary.json")
    p.add_argument("--snake-size-summary", default="../../snake_wm_v2/runs/vast_39636686_event_context1/hard_vs_prob_gap_comparison.json")
    p.add_argument("--pacman-evals", default="../../snake_wm_v2/runs/pacman_argmax_matrix/evals")
    return p.parse_args()


def load_json(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def tex(text: str) -> str:
    return text.replace("_", r"\_")


def snake_table(rows: list[dict], out: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Decoder & Layout & Hall. return & Real return & Real apples & Death & Win \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['decoder']} & {row['layout'].replace('_', ' ')} & "
            f"{row['hall_return']:.2f} & {row['real_return']:.2f} & "
            f"{row['real_apples']:.2f} & {row['death_rate']:.2f} & {row['win_rate']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write(out, "\n".join(lines) + "\n")


def snake_size_table(payload: dict, out: Path) -> None:
    rows = payload.get("rows", []) if payload else []
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Policy & Hard real apples & Prob. real apples & Hard gap & Prob. gap \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['policy']} & {row['hard_real_apples']:.2f} & {row['prob_real_apples']:.2f} & "
            f"{row['hard_gap']:.2f} & {row['prob_gap']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write(out, "\n".join(lines) + "\n")


def pacman_table(eval_root: Path, out: Path) -> list[dict]:
    rows = []
    for path in sorted(eval_root.glob("*/summary.json")):
        payload = load_json(path)
        if not payload:
            continue
        name = path.parent.name
        variant = name.split("_small_hard_")[0]
        rows.append({
            "variant": variant,
            "mode": payload["mode"],
            "return": payload["mean_return"],
            "steps": payload["mean_steps"],
            "pellets": payload.get("mean_pellets", payload.get("mean_predicted_pellets", 0.0)),
            "death": payload["death_rate"],
            "win": payload.get("win_rate"),
        })
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"World model & Eval & Return & Pellets & Steps & Death \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex(row['variant'])} & {tex(row['mode'])} & {row['return']:.2f} & {row['pellets']:.2f} & "
            f"{row['steps']:.1f} & {row['death']:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    write(out, "\n".join(lines) + "\n")
    return rows


def plot_pacman(rows: list[dict], out: Path) -> None:
    real = [r for r in rows if r["mode"] == "real"]
    if not real:
        return
    labels = [r["variant"] for r in real]
    pellets = [r["pellets"] for r in real]
    deaths = [r["death"] for r in real]
    plt.style.use("dark_background")
    fig, ax1 = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    fig.patch.set_facecolor("black")
    ax1.set_facecolor("black")
    ax1.bar(labels, pellets, color="#9bd85c", label="real pellets")
    ax1.set_ylabel("pellets collected")
    ax1.grid(axis="y", color="#243020", alpha=0.7)
    ax2 = ax1.twinx()
    ax2.plot(labels, deaths, color="#ff725f", marker="o", linewidth=2, label="death rate")
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_ylabel("death rate")
    ax1.set_title("Pac-Man transfer after argmax-event world-model training")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paper = Path(args.paper)
    tables = paper / "tables"
    figures = paper / "figures"

    snake_random = load_json(args.snake_random_summary)
    if snake_random:
        snake_table(snake_random, tables / "snake_argmax_randomized_summary.tex")

    snake_size = load_json(args.snake_size_summary)
    if snake_size:
        snake_size_table(snake_size, tables / "snake_hard_vs_prob_summary.tex")

    pac_rows = pacman_table(Path(args.pacman_evals), tables / "pacman_argmax_summary.tex")
    plot_pacman(pac_rows, figures / "pacman_argmax_transfer.png")
    print(f"wrote tables to {tables}")
    print(f"pacman rows: {len(pac_rows)}")


if __name__ == "__main__":
    main()
