# PushT Flow World Model Plan

## Scope

- Task: PushT only.
- Comparison target: native LeWM `wm_original_policy_original`.
- New run: `wm_flow_policy_original`.
- Policy/planner stays the original LeWM CEM/MPC eval pipeline.
- Only the world-model predictor is changed.
- Output root: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608`.

## Flow-WM Idea

The recommended first flow world model is an action-conditioned latent residual
Conditional Flow Matching model:

- Encoder: unchanged LeWM ViT encoder.
- Action encoder: unchanged LeWM `Embedder`.
- Projector and prediction projector: unchanged LeWM MLPs.
- Regularization: unchanged SIGReg term and weight.
- Dataloader: unchanged PushT Lance dataset and LeWM history/action framing.
- Predictor input: the same `z_t` history and embedded action history used by native LeWM.
- Flow target: residual dynamics `z_next - z_current`, not raw pixels and not a new policy.
- Training objective: flow-matching velocity MSE on the residual path.
- Inference: integrate the learned residual flow for 8 Euler steps, then return
  `z_current + residual`.

This keeps the experiment close to the original LeWM pipeline while testing the
main modeling hypothesis: replacing deterministic one-step latent regression
with a conditional generative dynamics model.

## Literature Rationale

- Flow Matching provides a simulation-free objective for continuous normalizing
  flows and is the base training objective for this WM variant:
  `https://arxiv.org/abs/2210.02747`.
- Recent world-model work increasingly uses generative latent dynamics rather
  than direct deterministic prediction when future states are multi-modal.
  Relevant examples include latent generative world models such as Diffusion
  Forcing (`https://arxiv.org/abs/2407.01392`) and latent-flow world-model
  systems (`https://arxiv.org/abs/2506.23434`).
- For this repo, residual CFM is preferred over pixel-space video diffusion
  because it preserves LeWM's lightweight latent planning interface and avoids
  changing the evaluator or policy stack.

## Pipeline Match Checklist

The flow-WM formal submission must match native LeWM except for the predictor:

- Uses `train.py`.
- Uses Hydra config `config/train/lewm.yaml`.
- Uses PushT data config `config/train/data/pusht.yaml`.
- Uses `pusht_expert_train.lance`.
- Uses `history_size=3`, `num_preds=1`, `img_size=224`, `embed_dim=192`.
- Uses `trainer.max_epochs=100`, `precision=bf16`, one GPU per job.
- Uses `optimizer=AdamW`, `lr=5e-5`, `weight_decay=1e-3`.
- Uses `loss.sigreg.weight=0.09`.
- Uses `JsonlMetricsCallback` and W&B logging.
- Uses `SaveCkptCallback` with epoch checkpoints.
- Uses automatic resume from Lightning `last.ckpt`.
- Uses project-storage runtime roots from `scripts/slurm_runtime_env.sh`.
- Uses stable dataloader settings from the native formal run:
  `NUM_WORKERS=6`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`,
  `PREFETCH_FACTOR=1`.

The only intended model config change is:

- Native: `config/train/model/lewm.yaml` -> `module.ARPredictor`.
- Flow WM: `config/train/model/lewm_flow.yaml` ->
  `module.ConditionalFlowPredictor`.

## Submitted Run Policy

- Submit seeds `0`, `1`, and `2`, matching the current native formal baseline.
- Preferred GPUs: H100, A100, H100.
- QOS: `embers`.
- Do not use `inferno`.
- Keep all logs, checkpoints, W&B cache, temporary files, and manifests under
  project storage, not `$HOME`.
- Supervisor behavior for this run:
  - keep resubmitting resumable WM chunks until `weights_epoch_100.pt` exists;
  - submit only original CEM eval after the flow WM checkpoint is ready;
  - do not train action-flow policy for this run.

## Expected Signals

Training metrics to compare against native LeWM:

- `fit/flow_loss` and `validate/flow_loss`.
- `fit/pred_loss` and `validate/pred_loss`, computed from sampled flow
  predictions for interpretability.
- `fit/loss` and `validate/loss`, including SIGReg.

Eval metrics:

- PushT real-environment success rate under original CEM.
- Per-episode successes.
- W&B run names, checkpoint path, and manifest path.

## Caveats

- Flow-WM inference is slower than deterministic AR prediction because every
  predicted latent requires multiple vector-field evaluations.
- With deterministic sampling from zero residual noise, the first run tests a
  stable mean-dynamics version of the flow. Stochastic or multi-sample CEM
  rollout can be added later if this version learns useful residual dynamics.
- This run is not action-flow policy training. It isolates the WM replacement.
