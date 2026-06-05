# Snake Hallucinated Worlds

This repository contains the code, configuration, paper source, generated figures, and lightweight result artifacts for a visual Snake world-model transfer study.

## Main result

Frozen action-conditioned visual world models can support PPO policy training, but imagined return and true-simulator return can diverge. The final 18-cell sweep and 3-iteration grounding summary are in `results/focused_v2/`.

## Public W&B report

https://wandb.ai/anothervibecoder-i-unemplyed/snake-hallucinated-worlds-v2/reports/Snake-Hallucinated-Worlds-v2--VmlldzoxNzEyNjYzMg==?accessToken=b9rqthrfzhibuu7brcnlh22jmwcc9dno4waru6ixk3po7rnzerbjx5obnb6pj7gi

## Build paper

```bash
cd paper
pdflatex main.tex
```

## Reproduce core pipeline

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m snake_wm_v2.run_focused_v2 --config configs/focused_v2.json --root runs/focused_v2 --wandb-mode online --skip-existing
```

Large datasets/checkpoints are intentionally excluded from Git. The included CSVs, tables, figures, and PDF are the lightweight reproducibility artifacts.
