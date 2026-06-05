# Snake WM v2: no-overlay hallucinated-world experiments

This is the cleaned v2 research pipeline for the paper question:

> Can a CNN policy learn Snake inside a learned visual world model, and transfer back to the true code simulator?

The v2 cleanup removes the main pilot confound: terminal states are no longer painted into RGB frames. Death/win are predicted through explicit scalar heads instead of a full-frame red/cyan/yellow tint.

## Design choices

- Observation: `128 x 128 x 3` RGB board-only frame.
- No HUD, no score text, no keyboard overlay.
- No death/win tint in the visual frame.
- Separate world-model heads: reward, done/status, and unclamped snake length.
- Hallucinated rollouts terminate from the `done_logit` head, not from visual color.
- Main metric: hallucinated return minus real-simulator return.
- W&B project: `snake-hallucinated-worlds-v2`.

## Quick local smoke run

```bash
cd snake_wm_v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m snake_wm_v2.generate_dataset --out runs/smoke_dataset --episodes 40 --max-transitions 800 --seed 123
python -m snake_wm_v2.train_world_model --dataset runs/smoke_dataset --out runs/smoke_wm --variant tiny --context 1 --steps 20 --batch-size 16 --wandb-mode disabled
python -m snake_wm_v2.train_policy --dataset runs/smoke_dataset --world-model runs/smoke_wm/latest.pt --out runs/smoke_policy --policy small --updates 2 --num-envs 4 --rollout-steps 8 --minibatch-size 16 --wandb-mode disabled
python -m snake_wm_v2.evaluate --policy runs/smoke_policy/latest.pt --episodes 5 --out runs/smoke_eval
python -m snake_wm_v2.make_figures --dataset runs/smoke_dataset --world-model runs/smoke_wm/latest.pt --out runs/smoke_figures
```

## Intended v2 experiment

The focused paper run is intentionally smaller than the pilot matrix:

- Dataset: 50k real transitions, no visual terminal overlay.
- World models: `wm_1m`, `wm_2m`.
- Contexts: `1`, `2`, `5` frames.
- Policies: `small`, `medium`, `large`.
- One frozen sweep policy per WM/context/policy combination.
- Grounding repeat for the strongest clean setting, using policy-induced real rollouts.

## Cloud workflow

Use Vast for debugging and main training, then port only the cleaned, working repo to GitHub.

```bash
export WANDB_MODE=online
export WANDB_ENTITY=anothervibecoder-i-unemplyed
export WANDB_PROJECT=snake-hallucinated-worlds-v2
python -m snake_wm_v2.run_focused_v2 --config configs/focused_v2.json
```

## Release rule

Do not publish old pilot artifacts as final evidence. The old tinted-frame runs can be mentioned only as a pilot failure mode. The clean GitHub repo should contain this v2 code, README, config files, paper source, and selected reproducibility artifacts after the v2 run finishes.

## Paper assets

After Vast artifacts are synced, export tables and figures for the paper:

```bash
python -m snake_wm_v2.export_paper_assets
```

Then build `paper/main.tex`. The paper links the W&B project at:

<https://wandb.ai/anothervibecoder-i-unemplyed/snake-hallucinated-worlds-v2>
