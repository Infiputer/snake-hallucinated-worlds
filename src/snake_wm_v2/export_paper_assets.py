from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_URL = "https://wandb.ai/anothervibecoder-i-unemplyed/snake-hallucinated-worlds-v2"
FPS = 30.0
GROUNDING_POLICY_UPDATES = 300
GROUNDING_NUM_ENVS = 32
GROUNDING_ROLLOUT_STEPS = 64
GROUNDING_IMAGINED_TRANSITIONS = GROUNDING_POLICY_UPDATES * GROUNDING_NUM_ENVS * GROUNDING_ROLLOUT_STEPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export synced results into paper tables and figures")
    p.add_argument("--sync-root", default="runs/vast_39487331_sync/runs/focused_v2")
    p.add_argument("--ablation-root", default="runs/vast_39541409_reward_ablation")
    p.add_argument("--matched-root", default="runs/matched_objective_sweep")
    p.add_argument("--paper-dir", default="paper")
    p.add_argument("--max-rows", type=int, default=24)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def row_sort_key(r: dict[str, str]) -> tuple[int, int, int]:
    wm_order = {"wm_1m": 0, "wm_2m": 1}.get(r.get("world_model", ""), 99)
    policy_order = {"small": 0, "medium": 1, "large": 2}.get(r.get("policy", ""), 99)
    try:
        context = int(r.get("context", 0))
    except Exception:
        context = 0
    return (wm_order, context, policy_order)


def write_objective_sweep_table(rows: list[dict[str, str]], objective: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = sorted([r for r in rows if r.get("objective") == objective], key=row_sort_key)
    if not selected:
        path.write_text("\\begin{tabular}{lllrrrr}\\toprule\nM & Ctx & CNN & Hall. & Real & Gap & Apples \\\\\n\\midrule\n\\multicolumn{7}{c}{Matched objective sweep pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    lines = ["\\begin{tabular}{lllrrrr}", "\\toprule", "M & Ctx & CNN & Hall. & Real & Gap & Apples \\\\", "\\midrule"]
    for r in selected:
        lines.append(f"{tex_escape(r['world_model'])} & {r['context']} & {tex_escape(r['policy'])} & {fmt(r['hallucinated_return'])} & {fmt(r['real_return'])} & {fmt(r['gap'])} & {fmt(r['real_apples'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_objective_comparison_table(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    reward = {(r["world_model"], r["context"], r["policy"]): r for r in rows if r.get("objective") == "wm"}
    length = {(r["world_model"], r["context"], r["policy"]): r for r in rows if r.get("objective") == "length_delta"}
    keys = sorted([k for k in reward if k in length], key=lambda k: row_sort_key({"world_model": k[0], "context": k[1], "policy": k[2]}))
    if not keys:
        path.write_text("\\begin{tabular}{lllrrr}\\toprule\nM & Ctx & CNN & Reward apples & Length apples & Delta \\\\\n\\midrule\n\\multicolumn{6}{c}{Matched objective comparison pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    lines = ["\\begin{tabular}{lllrrr}", "\\toprule", "M & Ctx & CNN & Reward apples & Length apples & Delta \\\\", "\\midrule"]
    for key in keys:
        rw = float(reward[key].get("real_apples", 0.0))
        ln = float(length[key].get("real_apples", 0.0))
        lines.append(f"{tex_escape(key[0])} & {key[1]} & {tex_escape(key[2])} & {fmt(rw)} & {fmt(ln)} & {fmt(rw - ln)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_grounding_table(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\\begin{tabular}{rrrrrrrr}\\toprule\nIter. & New trans. & Real min & Imag. trans. & Imag./real & Hall. & Real & Gap \\\\\n\\midrule\n\\multicolumn{8}{c}{Grounding results pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    lines = ["\\begin{tabular}{rrrrrrrr}", "\\toprule", "Iter. & New trans. & Real min & Imag. trans. & Imag./real & Hall. & Real & Gap \\\\", "\\midrule"]
    for r in rows:
        collected = float(r["collected_transitions"])
        real_minutes = collected / (FPS * 60.0)
        ratio = GROUNDING_IMAGINED_TRANSITIONS / max(collected, 1.0)
        lines.append(f"{r['iteration']} & {r['collected_transitions']} & {fmt(real_minutes)} & {GROUNDING_IMAGINED_TRANSITIONS} & {fmt(ratio, 1)} & {fmt(r['hallucinated_return'])} & {fmt(r['real_return'])} & {fmt(r['gap'])} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scalar_consistency_table(summary: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not summary:
        path.write_text("\\begin{tabular}{lrr}\\toprule\nMetric & Length head & Reward-integrated \\\\\n\\midrule\n\\multicolumn{3}{c}{Scalar diagnostic pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    rows = [
        ("Step MAE", summary.get("direct_length_step_mae", ""), summary.get("reward_integrated_step_mae", "")),
        ("Step median AE", summary.get("direct_length_step_median_ae", ""), summary.get("reward_integrated_step_median_ae", "")),
        ("Apple-step MAE", summary.get("direct_length_apple_step_mae", ""), summary.get("reward_integrated_apple_step_mae", "")),
        ("Better step fraction", summary.get("direct_length_better_step_fraction", ""), summary.get("reward_integrated_better_step_fraction", "")),
        ("Better episode count", summary.get("direct_length_better_episode_count", ""), summary.get("reward_integrated_better_episode_count", "")),
    ]
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Metric & Length head & Reward-integrated \\\\", "\\midrule"]
    for name, direct, reward in rows:
        lines.append(f"{tex_escape(name)} & {fmt(direct)} & {fmt(reward)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reward_objective_table(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\\begin{tabular}{lrrrr}\\toprule\nObjective & Hall. & Real & Apples & Death rate \\\\\n\\midrule\n\\multicolumn{5}{c}{PPO reward-objective ablation pending.} \\\\\n\\bottomrule\n\\end{tabular}\n", encoding="utf-8")
        return
    labels = {
        "wm": "Predicted reward",
        "length_delta": "Length change",
        "mixed": "Mixed",
    }
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Objective & Hall. & Real & Apples & Death rate \\\\", "\\midrule"]
    for r in rows:
        label = labels.get(r.get("reward_mode", ""), r.get("reward_mode", ""))
        lines.append(
            f"{tex_escape(label)} & {fmt(r.get('hallucinated_reward_return', ''))} & {fmt(r.get('real_return', ''))} & {fmt(r.get('real_apples', ''))} & {fmt(r.get('real_death_rate', ''))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def copy_figures(sync_root: Path, paper_dir: Path, ablation_root: Path, matched_root: Path) -> None:
    out = paper_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    required = [
        "snake_simulator_frame.png",
        "snake_environment_sequence.png",
        "hallucinated_vs_real.png",
        "wm_vs_sim_rollout.png",
        "wm_drift_curve.png",
        "scalar_consistency_mae.png",
        "scalar_consistency_rollout.png",
        "reward_objective_ablation.png",
        "matched_objective_context_scaling.png",
        "matched_objective_policy_scaling.png",
        "matched_objective_delta_heatmap.png",
        "matched_objective_scatter.png",
    ]
    candidates: list[tuple[Path, str]] = [
        (sync_root / "figures" / "snake_simulator_frame.png", "snake_simulator_frame.png"),
        (sync_root / "figures" / "snake_no_overlay_frame.png", "snake_simulator_frame.png"),
        (sync_root / "figures" / "snake_environment_sequence.png", "snake_environment_sequence.png"),
        (sync_root / "figures" / "hallucinated_vs_real.png", "hallucinated_vs_real.png"),
        (sync_root / "figures" / "hallucinated_vs_real_v2.png", "hallucinated_vs_real.png"),
        (sync_root / "figures" / "wm_vs_sim_rollout.png", "wm_vs_sim_rollout.png"),
        (sync_root / "figures" / "wm_vs_sim_rollout_v2.png", "wm_vs_sim_rollout.png"),
        (sync_root / "figures" / "wm_drift_curve.png", "wm_drift_curve.png"),
        (sync_root / "figures" / "wm_drift_curve_v2.png", "wm_drift_curve.png"),
        (ablation_root / "figures" / "scalar_consistency_mae.png", "scalar_consistency_mae.png"),
        (ablation_root / "figures" / "scalar_consistency_rollout.png", "scalar_consistency_rollout.png"),
        (ablation_root / "figures" / "reward_objective_ablation.png", "reward_objective_ablation.png"),
        (matched_root / "figures" / "matched_objective_context_scaling.png", "matched_objective_context_scaling.png"),
        (matched_root / "figures" / "matched_objective_policy_scaling.png", "matched_objective_policy_scaling.png"),
        (matched_root / "figures" / "matched_objective_delta_heatmap.png", "matched_objective_delta_heatmap.png"),
        (matched_root / "figures" / "matched_objective_scatter.png", "matched_objective_scatter.png"),
    ]
    if ablation_root != Path("runs/reward_length_ablation"):
        candidates.extend(
            [
                (Path("runs/reward_length_ablation") / "figures" / "scalar_consistency_mae.png", "scalar_consistency_mae.png"),
                (Path("runs/reward_length_ablation") / "figures" / "scalar_consistency_rollout.png", "scalar_consistency_rollout.png"),
                (Path("runs/reward_length_ablation") / "figures" / "reward_objective_ablation.png", "reward_objective_ablation.png"),
            ]
        )
    fallback = Path("runs/smoke_figures")
    for name in ("snake_simulator_frame.png", "snake_environment_sequence.png", "wm_vs_sim_rollout.png", "wm_drift_curve.png"):
        candidates.append((fallback / name, name))
    seen = set()
    for src, dst_name in candidates:
        if dst_name in seen or not src.exists():
            continue
        shutil.copy2(src, out / dst_name)
        seen.add(dst_name)
    aliases = {
        "snake_simulator_frame.png": "snake_no_overlay_frame.png",
        "hallucinated_vs_real.png": "hallucinated_vs_real_v2.png",
        "wm_vs_sim_rollout.png": "wm_vs_sim_rollout_v2.png",
        "wm_drift_curve.png": "wm_drift_curve_v2.png",
    }
    for old_name in aliases.values():
        old_path = out / old_name
        if old_path.exists():
            old_path.unlink()
    for new_name, old_name in aliases.items():
        new_path = out / new_name
        old_path = out / old_name
        if not new_path.exists() and old_path.exists():
            shutil.copy2(old_path, new_path)
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
    ablation_root = Path(args.ablation_root)
    matched_root = Path(args.matched_root)
    paper_dir = Path(args.paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(sync_root / "eval_summary.csv")
    grounding = read_csv(sync_root / "grounding" / "grounding_summary.csv")
    scalar_summary = read_json(ablation_root / "scalar_consistency_summary.json")
    reward_ablation = read_csv(ablation_root / "reward_objective_ablation.csv")
    matched_rows = read_csv(matched_root / "objective_eval_summary.csv")
    write_main_table(rows, paper_dir / "tables" / "main_eval_summary.tex", args.max_rows)
    write_objective_sweep_table(matched_rows, "wm", paper_dir / "tables" / "matched_reward_objective_summary.tex")
    write_objective_sweep_table(matched_rows, "length_delta", paper_dir / "tables" / "matched_length_objective_summary.tex")
    write_objective_comparison_table(matched_rows, paper_dir / "tables" / "matched_objective_comparison.tex")
    write_grounding_table(grounding, paper_dir / "tables" / "grounding_summary.tex")
    write_scalar_consistency_table(scalar_summary, paper_dir / "tables" / "scalar_consistency_summary.tex")
    write_reward_objective_table(reward_ablation, paper_dir / "tables" / "reward_objective_ablation.tex")
    copy_figures(sync_root, paper_dir, ablation_root, matched_root)
    (paper_dir / "wandb_project.txt").write_text(PROJECT_URL + "\n", encoding="utf-8")
    print(f"exported paper assets from {sync_root} and {ablation_root} to {paper_dir}")


if __name__ == "__main__":
    main()
