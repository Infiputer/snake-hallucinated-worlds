from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ENTITY = "anothervibecoder-i-unemplyed"
DEFAULT_PROJECT = "snake-hallucinated-worlds-v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a shareable W&B report for v2 results")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--eval-summary", default="runs/vast_39487331_sync/runs/focused_v2/eval_summary.csv")
    p.add_argument("--grounding-summary", default="runs/vast_39487331_sync/runs/focused_v2/grounding/grounding_summary.csv")
    p.add_argument("--out", default="paper/wandb_report_url.txt")
    p.add_argument("--title", default="Snake Hallucinated Worlds v2")
    p.add_argument("--enable-share-link", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: str, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return x


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
            vals.append(fmt(val) if col in {"hallucinated_return", "real_return", "gap", "real_apples", "real_death_rate", "collected_transitions", "total_policy_collected_transitions"} else val)
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    try:
        import wandb_workspaces.reports.v2 as wr
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install with `pip install wandb-workspaces` before publishing the report.") from exc

    eval_rows = sorted(read_rows(Path(args.eval_summary)), key=lambda r: float(r.get("gap", 0.0)), reverse=True)
    grounding_rows = read_rows(Path(args.grounding_summary))
    project_url = f"https://wandb.ai/{args.entity}/{args.project}"
    best = eval_rows[0] if eval_rows else None
    description = "CNN policies trained inside action-conditioned visual Snake world models, then evaluated in the real simulator."
    blocks = [
        wr.H1("Snake hallucinated-world v2"),
        wr.P(description),
        wr.P(f"Project: {project_url}"),
        wr.H2("Main transfer table"),
        wr.P(markdown_table(eval_rows, ["world_model", "context", "policy", "hallucinated_return", "real_return", "gap", "real_apples", "real_death_rate"])),
    ]
    if best:
        blocks.extend([
            wr.H2("Highest current hallucinated-real gap"),
            wr.P(
                f"`{best['world_model']}` context `{best['context']}` policy `{best['policy']}`: "
                f"hallucinated return {fmt(best['hallucinated_return'])}, real return {fmt(best['real_return'])}, gap {fmt(best['gap'])}."
            ),
        ])
    blocks.extend([
        wr.H2("Grounding loop"),
        wr.P(markdown_table(grounding_rows, ["iteration", "collected_transitions", "total_policy_collected_transitions", "hallucinated_return", "real_return", "gap"])),
        wr.H2("Notes"),
        wr.P("Terminal state is not painted into RGB frames. The world model predicts terminal probability, reward, and unclamped floating-point snake length through scalar heads."),
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
