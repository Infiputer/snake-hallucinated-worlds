# Snake hallucinated worlds

This repository contains the paper source and runnable implementation for the Snake hallucinated-world experiments.

Research question:

> Can a CNN policy learn Snake inside a learned visual world model, and does that behavior transfer back to the true simulator?

## Repository layout

- `implementation/`: clean runnable package, web UI, screenshots, and the four command pipeline.
- `paper/`: LaTeX source, figures, tables, and compiled draft PDF for the main paper.
- `results/` and selected `runs/`: small reproducibility summaries used by the paper.

Start with [implementation/README.md](implementation/README.md) for the current code path.

## Current implementation

The current world model predicts the next RGB frame plus two discrete events: apple eaten and snake death. The web UI can start the learned simulator from custom apple and rock arrangements.

Run locally:

```bash
cd implementation
python -m pip install -r requirements.txt
python -m pip install -e .
snake-inference --checkpoint runs/wm_5m_random_layout/latest.pt --location localhost:8055
```

Open `http://localhost:8055`.

## Four command pipeline

```bash
snake-generate-data      # generate randomized-layout simulator data
snake-train-wm           # train the event world model
snake-inference          # run inference and the web UI
snake-train-cnn-agent    # train a CNN PPO policy inside the WM
```

See [implementation/README.md](implementation/README.md) for arguments and examples.

## Paper

The main paper draft is in [paper/main.tex](paper/main.tex), with the current compiled draft at [paper/main.pdf](paper/main.pdf).
