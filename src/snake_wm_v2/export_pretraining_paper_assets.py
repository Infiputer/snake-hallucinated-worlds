from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Paper 2 fixed-WM pretraining tables and figures")
    p.add_argument("--root", default="runs/pretraining_study")
    p.add_argument("--paper", default="paper_pretraining")
    return p.parse_args()


def fmt(x: str | float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary_table(rows: list[dict[str, str]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Final return & Best return & Return AUC & Real steps \\",
        r"\midrule",
    ]
    for row in rows:
        cond = row["condition"].replace("_", r"\_")
        curve = read_rows(Path(row["run_dir"]) / "eval_curve.csv")
        best_return = max(float(r["eval/return_mean"]) for r in curve)
        lines.append(
            f"{cond} & {fmt(row['final_return'])} & {fmt(best_return)} & {fmt(row['return_auc'], 1)} & {int(float(row['env_steps']))} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    paper = Path(args.paper)
    fig_out = paper / "figures"
    tab_out = paper / "tables"
    fig_out.mkdir(parents=True, exist_ok=True)
    tab_out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(root / "summary.csv")
    write_summary_table(rows, tab_out / "pretraining_summary.tex")
    for name in ["real_sample_efficiency.png", "return_auc.png"]:
        src = root / "figures" / name
        if src.exists():
            shutil.copy2(src, fig_out / name)
    (paper / "experiment_summary.csv").write_text((root / "summary.csv").read_text(encoding="utf-8"), encoding="utf-8")
    print(f"exported Paper 2 assets from {root} to {paper}")


if __name__ == "__main__":
    main()
