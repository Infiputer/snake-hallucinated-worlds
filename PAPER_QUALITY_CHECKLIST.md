# Paper quality checklist

Use this checklist before treating either paper as final.

## Framing

- State claims cautiously: exploratory Snake results, not broad robotics or open-world claims.
- Prefer "transfer gap", "hallucination-real gap", or "model-exploitation gap" over "reward hacking" unless the evidence specifically proves reward-model exploitation.
- Keep negative results, but present them as scientific outcomes, not infrastructure or implementation failures.
- Do not include OOMs, debugging notes, cloud-provider details, admin notes, or final-submission TODOs in the paper body.

## Baselines and statistics

- Report random/heuristic/direct-PPO baselines where available; clearly label missing or incomplete baselines.
- Prefer learning curves and real-simulator sample efficiency over raw PPO losses.
- Include apples, returns, death rate, and transfer gap.
- When enough repetitions exist, report mean, standard error or bootstrap intervals, and episode-level distributions.
- If a cell is single-seed, say so and avoid trend claims.

## Methods

- Include exact Snake environment rules, reward function, action handling, max episode length, and evaluation protocol.
- Include dataset policy mixture, transition count, split/seeds when available, and dataset summary statistics.
- Include world-model architecture, loss terms, optimizer, batch size, context length, and scalar heads.
- Include PPO rollout length, env count, learning rate, entropy coefficient, clip range, discount, GAE lambda, epochs, minibatch size, and deterministic evaluation details.

## Figures and tables

- Use clear large figures over many tiny rollout grids.
- Label scatter/heatmap axes as transfer metrics, not reward-hacking metrics.
- Include real apples in main transfer tables where space allows.
- Move local artifact paths and long manifests to appendix or README.

## Paper 1-specific

- Main message: frozen learned Snake worlds can create transfer gaps; on-policy grounding can reduce the gap in selected settings, but transfer remains unstable.
- Grounding improvements should not be over-attributed unless ablations isolate world-model fine-tuning, policy warm-starting, replay, and additional PPO.

## Paper 2-specific

- Main question: does fixed-WM policy pretraining reduce real-simulator PPO sample cost?
- Compare policies at equal real-simulator step budgets and with AUC over real-return learning curves.
- Do not compare PPO losses as the primary result; they are not calibrated across initializations.
- Frame the current first experiment as a negative result unless further seeds/baselines change the conclusion.
