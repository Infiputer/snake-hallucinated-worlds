#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install -e . -r requirements.txt >/dev/null
python3 -m snake_wm_v2.generate_dataset --out runs/smoke_dataset --episodes 40 --max-transitions 800 --seed 123
python3 -m snake_wm_v2.train_world_model --dataset runs/smoke_dataset --out runs/smoke_wm --variant tiny --context 1 --steps 20 --batch-size 16 --wandb-mode disabled
python3 -m snake_wm_v2.train_policy --dataset runs/smoke_dataset --world-model runs/smoke_wm/latest.pt --out runs/smoke_policy --policy small --updates 2 --num-envs 4 --rollout-steps 8 --minibatch-size 16 --wandb-mode disabled
python3 -m snake_wm_v2.evaluate --policy runs/smoke_policy/latest.pt --episodes 5 --out runs/smoke_eval
python3 -m snake_wm_v2.make_figures --dataset runs/smoke_dataset --world-model runs/smoke_wm/latest.pt --out runs/smoke_figures
