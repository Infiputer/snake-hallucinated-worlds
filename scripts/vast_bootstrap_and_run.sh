#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SnakeGame/snake_wm_v2
python3 -m pip install --upgrade pip
python3 -m pip install -e . -r requirements.txt
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-anothervibecoder-i-unemplyed}
export WANDB_PROJECT=${WANDB_PROJECT:-snake-hallucinated-worlds-v2}
python3 -m snake_wm_v2.run_focused_v2 --config configs/focused_v2.json --root runs/focused_v2 --wandb-mode online --skip-existing
