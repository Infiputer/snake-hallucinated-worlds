#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/src"
export WANDB_PROJECT="${WANDB_PROJECT:-snake-hallucinated-worlds-pacman}"

DATASET="${DATASET:-runs/datasets/pacman_random_30k}"
OUT_ROOT="${OUT_ROOT:-runs/pacman_argmax_matrix}"
SEED="${SEED:-20260606}"
WM_STEPS_TINY="${WM_STEPS_TINY:-8000}"
WM_STEPS_1M="${WM_STEPS_1M:-12000}"
WM_STEPS_2M="${WM_STEPS_2M:-16000}"
POLICY_UPDATES="${POLICY_UPDATES:-180}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"

mkdir -p "$OUT_ROOT"

wm_done() {
  local wm_dir="$1"
  local target_steps="$2"
  python3 - "$wm_dir" "$target_steps" <<'PY'
import json
import sys
from pathlib import Path

wm_dir = Path(sys.argv[1])
target = int(sys.argv[2])
latest = wm_dir / "latest.pt"
val = wm_dir / "latest_val.json"
if not latest.exists() or not val.exists():
    raise SystemExit(1)
try:
    step = int(json.loads(val.read_text()).get("step", 0))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if step >= target else 1)
PY
}

if [ ! -f "$DATASET/dataset_meta.json" ]; then
  python3 -m snake_wm_v2.generate_pacman_dataset \
    --episodes 1200 \
    --max-steps 512 \
    --max-transitions 30000 \
    --random-map \
    --seed "$SEED" \
    --out "$DATASET"
fi

run_cell() {
  local variant="$1"
  local wm_steps="$2"
  local wm_dir="$OUT_ROOT/world_models/${variant}"
  local policy_dir="$OUT_ROOT/policies/${variant}_small_hard"
  local hall_eval="$OUT_ROOT/evals/${variant}_small_hard_hall"
  local real_eval="$OUT_ROOT/evals/${variant}_small_hard_real_random"

  if ! wm_done "$wm_dir" "$wm_steps"; then
    python3 -m snake_wm_v2.train_event_world_model \
      --dataset "$DATASET" \
      --out "$wm_dir" \
      --variant "$variant" \
      --steps "$wm_steps" \
      --batch-size 16 \
      --num-workers 2 \
      --save-every 2000 \
      --val-every 1000 \
      --seed "$SEED" \
      --wandb-mode auto
  fi

  if [ ! -f "$policy_dir/latest.pt" ]; then
    python3 -m snake_wm_v2.train_event_policy \
      --world-model "$wm_dir/latest.pt" \
      --dataset "$DATASET" \
      --out "$policy_dir" \
      --policy small \
      --updates "$POLICY_UPDATES" \
      --num-envs 16 \
      --rollout-steps 48 \
      --epochs 4 \
      --minibatch-size 192 \
      --reward-decoder hard \
      --max-episode-steps 256 \
      --seed "$SEED" \
      --wandb-mode auto
  fi

  python3 -m snake_wm_v2.evaluate_pacman_policy \
    --policy "$policy_dir/latest.pt" \
    --out "$hall_eval" \
    --mode hallucinated \
    --world-model "$wm_dir/latest.pt" \
    --dataset "$DATASET" \
    --episodes "$EVAL_EPISODES" \
    --max-steps 256 \
    --reward-decoder hard \
    --seed "$SEED"

  python3 -m snake_wm_v2.evaluate_pacman_policy \
    --policy "$policy_dir/latest.pt" \
    --out "$real_eval" \
    --mode real \
    --episodes "$EVAL_EPISODES" \
    --max-steps 512 \
    --random-map \
    --seed "$SEED"
}

run_cell tiny "$WM_STEPS_TINY"
run_cell wm_1m "$WM_STEPS_1M"
run_cell wm_2m "$WM_STEPS_2M"

python3 - <<'PY'
import json
from pathlib import Path

root = Path("runs/pacman_argmax_matrix/evals")
rows = []
for path in sorted(root.glob("*/summary.json")):
    payload = json.loads(path.read_text())
    rows.append({
        "name": path.parent.name,
        "mode": payload["mode"],
        "mean_return": payload["mean_return"],
        "mean_steps": payload["mean_steps"],
        "pellets": payload.get("mean_pellets", payload.get("mean_predicted_pellets")),
        "death_rate": payload["death_rate"],
        "win_rate": payload.get("win_rate"),
    })
print(json.dumps(rows, indent=2))
PY

python3 -m snake_wm_v2.export_argmax_paper_assets \
  --paper papers \
  --pacman-evals "../../snake_wm_v2/runs/pacman_argmax_matrix/evals"
