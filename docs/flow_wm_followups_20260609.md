# PushT Flow-WM Follow-Ups, 2026-06-09

## Strict Pipeline Rule

All formal runs in this follow-up keep the native LeWM PushT pipeline intact
unless the row explicitly states a diagnostic override.

- Task: PushT only.
- Data: official PushT Lance dataset through `config/train/data/pusht.yaml`.
- Training entrypoint: `train.py` with Hydra config `config/train/lewm.yaml`.
- Eval entrypoint: `eval.py` with Hydra config `config/eval/pusht.yaml`.
- Scheduler: existing train/eval/supervisor Slurm wrappers.
- Planner/policy for the formal WM comparison: original LeWM CEM/MPC.
- Encoder, projector, action encoder, prediction projector, SIGReg, optimizer,
  checkpointing, resume logic, W&B logging, and output layout remain unchanged.
- Outputs stay under project storage:
  `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/`.
- QOS: `embers`; do not use `inferno`.

## Follow-Up 1: Cost-Ranking Diagnostic

Goal: test whether the learned world model ranks the dataset expert action
chunk below zero/random action chunks under the same LeWM cost function used by
CEM.

Reason: current flow-WM checkpoints reduce `validate/flow_loss`, but real PushT
success does not improve. A ranking diagnostic directly checks whether the
model cost surface is useful for the original planner.

Implementation:

- New script: `diagnose_cost_ranking.py`.
- It uses `config/eval/pusht.yaml`, the same dataset loader, image transform,
  action/state scalers, `WorldModelPolicy._prepare_info`, and
  `model.get_cost(...)` path as real eval.
- It logs JSONL metrics and W&B metrics:
  `diag/mean_expert_cost`, `diag/mean_zero_cost`,
  `diag/mean_random_cost`, `diag/mean_expert_rank`, and
  `diag/mean_random_better_frac`.
- It is not a replacement for real-environment eval; it is a fast diagnostic to
  explain why CEM succeeds or fails.

Planned comparisons:

| Run | Checkpoint family | Purpose |
| --- | --- | --- |
| Native LeWM | `pusht_native_formal_20260605` | Positive control; expert actions should rank well. |
| Flow WM | `pusht_flow_wm_formal_20260608` | Diagnose whether low flow loss gives a useful planner cost. |

## Follow-Up 2: Endpoint-Aligned Flow WM

Goal: keep the residual conditional flow WM, but add a small endpoint prediction
loss so the same deterministic rollout used by CEM is trained to land near the
next latent embedding.

Reason: the first flow-WM run optimizes only the residual flow-matching velocity
objective. The original LeWM planner consumes deterministic next-latent
predictions through `model.predict(...)`; if the velocity objective is not
aligned with that endpoint, CEM can receive a poor cost surface even while
`validate/flow_loss` improves.

Formal run change:

- Model: `model=lewm_flow`.
- Variant label: `wm_variant=flow_endpoint`.
- Added override: `loss.flow_pred.weight=0.1`.
- New logged metric alias: `fit/endpoint_loss` and `validate/endpoint_loss`.
- Everything else follows the strict pipeline rule above.

This run should be compared against:

- Native LeWM full-CEM PushT eval: current best observed success is 80-88%.
- Original flow-WM full-CEM PushT eval: current observed success is 0-10%.

Success signal:

- During training: endpoint/pred loss should drop substantially below the
  original flow-WM value near 1.2 while flow loss remains stable.
- During eval: original CEM real-environment success should improve over the
  original flow-WM run.
