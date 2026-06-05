from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_ENTITY = "anothervibecoder-i-unemplyed"
DEFAULT_PROJECT = "snake-wm-pretraining-v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a shareable W&B report for the fixed-WM pretraining study")
    p.add_argument("--entity", default=DEFAULT_ENTITY)
    p.add_argument("--project", default=DEFAULT_PROJECT)
    p.add_argument("--summary", default="runs/pretraining_study/summary.csv")
    p.add_argument("--out", default="paper_pretraining/wandb_report_url.txt")
    p.add_argument("--title", default="Snake Fixed World-Model Pretraining")
    p.add_argument("--enable-share-link", action="store_true")
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x: str, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return x


def markdown_table(rows: list[dict[str, str]]) -> str:
    cols = ["condition", "final_return", "final_apples", "final_death_rate", "return_auc", "env_steps"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = []
    for row in rows:
        vals = [fmt(row.get(c, ""), 1 if c == "return_auc" else 3) if c != "condition" else row.get(c, "") for c in cols]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    try:
        import wandb_workspaces.reports.v2 as wr
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: install with `pip install wandb-workspaces` before publishing the report.") from exc
    rows = read_rows(Path(args.summary))
    project_url = f"https://wandb.ai/{args.entity}/{args.project}"
    description = "Fixed Snake world-model PPO pretraining, followed by PPO fine-tuning in the true simulator."
    blocks = [
        wr.H1("Snake fixed world-model pretraining"),
        wr.P(description),
        wr.P(f"Project: {project_url}"),
        wr.H2("Question"),
        wr.P("Does policy optimization inside a fixed learned visual simulator reduce the number of true-simulator steps needed by PPO?"),
        wr.H2("Result summary"),
        wr.P(markdown_table(rows)),
        wr.H2("Interpretation"),
        wr.P("In this first run, world-model pretraining did not improve real-simulator sample efficiency. The scratch PPO baseline reached a higher peak return and a larger area under the real-return curve."),
    ]
    report = wr.Report(entity=args.entity, project=args.project, title=args.title, description=description)
    report.blocks = blocks
    report.save()
    share_url = report.enable_share_link() if args.enable_share_link else report.url
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(share_url + "\n", encoding="utf-8")
    out.with_suffix(".tex").write_text(r"\href{" + share_url + r"}{W\&B report}" + "\n", encoding="utf-8")
    print(share_url)


if __name__ == "__main__":
    main()
