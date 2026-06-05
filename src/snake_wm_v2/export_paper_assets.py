from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_URL = "https://wandb.ai/anothervibecoder-i-unemplyed/snake-hallucinated-worlds-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export synced v2 results into paper tables and figures")
    p.add_argument("--sync-root", default="runs/vast_39487331_sync/runs/focused_v2")
    p.add_argument("--paper-dir", default="paper")
    p.add_argument("--max-rows", type=int, default=24)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: str | float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def tex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_main_table(rows: list[dict[str, str]], path: Path, max_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\\begin{tabular}{llllrrr}\\toprule\nWM & Ctx & CNN & Hall. & Real & Gap & Apples \\\\\n\\midrule\n\\multicolumn{7}{c}{Results pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    rows = sorted(rows, key=lambda r: float(r.get("gap", 0.0)), reverse=True)[:max_rows]
    lines = ["\\begin{tabular}{lllrrrr}", "\\toprule", "WM & Ctx & CNN & Hall. & Real & Gap & Apples \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{tex_escape(r['world_model'])} & {r['context']} & {tex_escape(r['policy'])} & {fmt(r['hallucinated_return'])} & {fmt(r['real_return'])} & {fmt(r['gap'])} & {fmt(r['real_apples'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grounding_table(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\\begin{tabular}{rrrrrr}\\toprule\nIter. & New data & Total data & Hall. & Real & Gap \\\\\n\\midrule\n\\multicolumn{6}{c}{Grounding results pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    lines = ["\\begin{tabular}{rrrrrr}", "\\toprule", "Iter. & New data & Total data & Hall. & Real & Gap \\\\", "\\midrule"]
    for r in rows:
        lines.append(f"{r['iteration']} & {r['collected_transitions']} & {r['total_policy_collected_transitions']} & {fmt(r['hallucinated_return'])} & {fmt(r['real_return'])} & {fmt(r['gap'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_figures(sync_root: Path, paper_dir: Path) -> None:
    out = paper_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    required = [
        "snake_no_overlay_frame.png",
        "hallucinated_vs_real_v2.png",
        "wm_vs_sim_rollout_v2.png",
        "wm_drift_curve_v2.png",
    ]
    candidates = [
        sync_root / "figures" / "snake_no_overlay_frame.png",
        sync_root / "figures" / "hallucinated_vs_real_v2.png",
        sync_root / "figures" / "wm_vs_sim_rollout_v2.png",
        sync_root / "figures" / "wm_drift_curve_v2.png",
    ]
    fallback = Path("runs/smoke_figures")
    for name in ("snake_no_overlay_frame.png", "wm_vs_sim_rollout_v2.png", "wm_drift_curve_v2.png"):
        candidates.append(fallback / name)
    seen = set()
    for src in candidates:
        if src.name in seen or not src.exists():
            continue
        shutil.copy2(src, out / src.name)
        seen.add(src.name)
    for name in required:
        dst = out / name
        if dst.exists():
            continue
        img = Image.new("RGB", (1000, 500), (245, 245, 245))
        draw = ImageDraw.Draw(img)
        draw.rectangle((30, 30, 970, 470), outline=(120, 120, 120), width=3)
        draw.text((60, 220), f"{name}\nPending final Vast export", fill=(30, 30, 30))
        img.save(dst)


def main() -> None:
    args = parse_args()
    sync_root = Path(args.sync_root)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(sync_root / "eval_summary.csv")
    grounding = read_csv(sync_root / "grounding" / "grounding_summary.csv")
    write_main_table(rows, paper_dir / "tables" / "main_eval_summary.tex", args.max_rows)
    write_grounding_table(grounding, paper_dir / "tables" / "grounding_summary.tex")
    copy_figures(sync_root, paper_dir)
    (paper_dir / "wandb_project.txt").write_text(PROJECT_URL + "\n", encoding="utf-8")
    print(f"exported paper assets from {sync_root} to {paper_dir}")


if __name__ == "__main__":
    main()
