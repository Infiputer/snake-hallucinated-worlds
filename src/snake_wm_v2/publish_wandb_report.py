from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_ENTITY = "anothervibecoder-i-unemplyed"
DEFAULT_PROJECT = "snake-hallucinated-worlds-v2"
FPS = 30.0
GROUNDING_POLICY_UPDATES = 300
GROUNDING_NUM_ENVS = 32
GROUNDING_ROLLOUT_STEPS = 64
GROUNDING_IMAGINED_TRANSITIONS = GROUNDING_POLICY_UPDATES * GROUNDING_NUM_ENVS * GROUNDING_ROLLOUT_STEPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a shareable W&B report for Snake world-model results")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--eval-summary", default="runs/vast_39487331_sync/runs/focused_v2/eval_summary.csv")
    p.add_argument("--grounding-summary", default="runs/vast_39487331_sync/runs/focused_v2/grounding/grounding_summary.csv")
    p.add_argument("--scalar-summary", default="runs/vast_39541409_reward_ablation/scalar_consistency_summary.json")
    p.add_argument("--reward-objective-ablation", default="runs/vast_39541409_reward_ablation/reward_objective_ablation.csv")
    p.add_argument("--matched-objective-summary", default="runs/matched_objective_sweep/objective_eval_summary.csv")
    p.add_argument("--out", default="paper/wandb_report_url.txt")
    p.add_argument("--title", default="Snake Hallucinated Worlds")
    p.add_argument("--enable-share-link", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x: object, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def markdown_table(rows: list[dict[str, str]], cols: list[str], max_rows: int = 18) -> str:
    if not rows:
        return "_No rows available yet._"
    rows = rows[:max_rows]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            vals.append(fmt(val) if col in {"hallucinated_return", "real_return", "gap", "real_apples", "real_death_rate", "collected_transitions", "total_policy_collected_transitions", "hallucinated_reward_return", "hallucinated_done_rate", "real_mean_steps"} else val)
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def matched_summary_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_Matched objective sweep pending._"
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("objective", ""), []).append(row)
    lines = ["| objective | cells | mean real apples | mean real return | mean transfer gap |", "| --- | --- | --- | --- | --- |"]
    for objective in ("wm", "length_delta"):
        selected = grouped.get(objective, [])
        if not selected:
            continue
        cells = len(selected)
        apples = sum(float(r.get("real_apples", 0.0)) for r in selected) / cells
        real_return = sum(float(r.get("real_return", 0.0)) for r in selected) / cells
        gap = sum(float(r.get("gap", 0.0)) for r in selected) / cells
        label = "Predicted reward" if objective == "wm" else "Unclamped length delta"
        lines.append(f"| {label} | {cells} | {fmt(apples)} | {fmt(real_return)} | {fmt(gap)} |")
    return "\n".join(lines)


def matched_comparison_markdown(rows: list[dict[str, str]], max_rows: int = 18) -> str:
    reward = {(r.get("world_model", ""), r.get("context", ""), r.get("policy", "")): r for r in rows if r.get("objective") == "wm"}
    length = {(r.get("world_model", ""), r.get("context", ""), r.get("policy", "")): r for r in rows if r.get("objective") == "length_delta"}
    keys = [k for k in reward if k in length]
    if not keys:
        return "_Matched comparison pending._"
    policy_order = {"small": 0, "medium": 1, "large": 2}
    keys = sorted(keys, key=lambda k: (k[0], int(k[1]), policy_order.get(k[2], 99)))[:max_rows]
    lines = ["| world_model | context | policy | reward apples | length apples | delta |", "| --- | --- | --- | --- | --- | --- |"]
    for key in keys:
        rw = float(reward[key].get("real_apples", 0.0))
        ln = float(length[key].get("real_apples", 0.0))
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {fmt(rw)} | {fmt(ln)} | {fmt(rw - ln)} |")
    return "\n".join(lines)


def scalar_markdown(summary: dict[str, object]) -> str:
    if not summary:
        return "_No scalar diagnostic available yet._"
    rows = [
        ("Step MAE", summary.get("direct_length_step_mae", ""), summary.get("reward_integrated_step_mae", "")),
        ("Step median AE", summary.get("direct_length_step_median_ae", ""), summary.get("reward_integrated_step_median_ae", "")),
        ("Apple-step MAE", summary.get("direct_length_apple_step_mae", ""), summary.get("reward_integrated_apple_step_mae", "")),
        ("Better step fraction", summary.get("direct_length_better_step_fraction", ""), summary.get("reward_integrated_better_step_fraction", "")),
        ("Better episode count", summary.get("direct_length_better_episode_count", ""), summary.get("reward_integrated_better_episode_count", "")),
    ]
    lines = ["| Metric | Length head | 3 + cumulative predicted reward |", "| --- | --- | --- |"]
    for name, direct, reward in rows:
        lines.append(f"| {name} | {fmt(direct)} | {fmt(reward)} |")
    return "\n".join(lines)


def grounding_markdown(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No grounding rows available yet._"
    header = "| iteration | new real transitions | real minutes @30FPS | imagined PPO transitions | imagined/real | hallucinated return | real return | gap |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
    body = []
    for row in rows:
        collected = float(row.get("collected_transitions", 0.0))
        real_minutes = collected / (FPS * 60.0)
        ratio = GROUNDING_IMAGINED_TRANSITIONS / max(collected, 1.0)
        body.append(
            "| "
            + " | ".join(
                [
                    row.get("iteration", ""),
                    row.get("collected_transitions", ""),
                    fmt(real_minutes),
                    str(GROUNDING_IMAGINED_TRANSITIONS),
                    fmt(ratio, 1),
                    fmt(row.get("hallucinated_return", "")),
                    fmt(row.get("real_return", "")),
                    fmt(row.get("gap", "")),
                ]
            )
            + " |"
        )
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    try:
        import wandb_workspaces.reports.v2 as wr
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install with `pip install wandb-workspaces` before publishing the report.") from exc

    eval_rows = sorted(read_rows(Path(args.eval_summary)), key=lambda r: float(r.get("gap", 0.0)), reverse=True)
    grounding_rows = read_rows(Path(args.grounding_summary))
    scalar_summary = read_json(Path(args.scalar_summary))
    objective_rows = read_rows(Path(args.reward_objective_ablation))
    matched_rows = read_rows(Path(args.matched_objective_summary))
    project_url = f"https://wandb.ai/{args.entity}/{args.project}"
    best = eval_rows[0] if eval_rows else None
    description = "CNN policies trained inside action-conditioned visual Snake world models, then evaluated in the real simulator."
    blocks = [
        wr.H1("Snake hallucinated worlds"),
        wr.P(description),
        wr.P(f"Project: {project_url}"),
        wr.H2("Main transfer table"),
        wr.P(markdown_table(eval_rows, ["world_model", "context", "policy", "hallucinated_return", "real_return", "gap", "real_apples", "real_death_rate"])),
    ]
    if best:
        blocks.extend([
            wr.H2("Highest current transfer gap"),
            wr.P(
                f"`{best['world_model']}` context `{best['context']}` policy `{best['policy']}`: "
                f"hallucinated return {fmt(best['hallucinated_return'])}, real return {fmt(best['real_return'])}, gap {fmt(best['gap'])}."
            ),
        ])
    blocks.extend([
        wr.H2("Matched objective sweep"),
        wr.P("For every world-model size, context length, and CNN size, the matched sweep trains one policy on predicted scalar reward and one policy on the raw unclamped length-head delta. This is the apples-to-apples comparison requested for the paper tables."),
        wr.P(matched_summary_markdown(matched_rows)),
        wr.P(matched_comparison_markdown(matched_rows)),
        wr.H2("Grounding loop"),
        wr.P("Grounding is periodic, not continuous: collect real simulator transitions, fine-tune the existing world model, then freeze it while PPO trains the next policy. Real-data minutes below assume 30 FPS."),
        wr.P(grounding_markdown(grounding_rows)),
        wr.H2("Scalar consistency diagnostic"),
        wr.P("For Snake, length equals initial length plus the cumulative apple reward under the raw simulator reward. This diagnostic compares the direct length head with the length implied by cumulative predicted reward on fixed-world-model rollouts."),
        wr.P(scalar_markdown(scalar_summary)),
        wr.H2("PPO reward-objective ablation"),
        wr.P("The reward-objective ablation keeps the world model fixed and changes only the scalar optimized by PPO. This separates scalar prediction quality from the policy-transfer question."),
        wr.P(markdown_table(objective_rows, ["reward_mode", "policy", "hallucinated_reward_return", "real_return", "real_apples", "real_death_rate", "real_mean_steps"])),
        wr.H2("Notes"),
        wr.P("RGB frames contain only the game board. Death probability, reward, and unclamped floating-point snake length are predicted through separate scalar heads."),
    ])
    report = wr.Report(entity=args.entity, project=args.project, title=args.title, description=description)
    report.blocks = blocks
    report.save()
    share_url = report.enable_share_link() if args.enable_share_link else report.url
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(share_url + "\n", encoding="utf-8")
    print(share_url)


if __name__ == "__main__":
    main()
