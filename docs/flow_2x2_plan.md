# LeWM Flow 2x2 Experiment Plan

## Summary

- Branch: `exp/flow-2x2-pusht-cube`.
- Remote: `origin` points to `git@github.com:thedannyliu/le-wm.git`.
- Tasks: PushT and OGBench-Cube.
- Matrix:
  - World model: original deterministic LeWM predictor vs conditional latent flow predictor.
  - Policy/planner: original CEM/MPC vs conditional action-flow proposal scored by the active world model.
- Output root: `$STABLEWM_HOME/experiments/flow_2x2_20260604/`.
- Git tracks only source, configs, docs, manifests, and lightweight summaries. Checkpoints, logs, videos, W&B local files, and old root result files stay untracked.

## Implementation

- Keep the original LeWM encoder, action encoder, projector, prediction projector, and SIGReg pipeline intact.
- Add a flow world-model variant that replaces the one-step predictor with a conditional flow-matching predictor over next latent embeddings.
- Add a flow policy variant that samples action chunks conditioned on current and goal latents, then uses the active world model to score proposals.
- Add Hydra switches for task, seed, world-model variant, policy variant, output root, W&B metadata, and eval output naming.
- Add local run manifests with command, git SHA, config snapshot, checkpoint path, W&B identifiers, metrics file path, and generated video/result paths.

## Logging

- Log all training, validation, and real-environment eval runs to W&B.
- Default W&B project: `lewm-flow-2x2`.
- Default W&B entity: unset; use the active W&B login, or override `wandb.config.entity=<entity>`.
- Default W&B group: `{task}/{wm_variant}/{policy_variant}`.
- Default W&B name: `{task}_{wm_variant}_{policy_variant}_seed{seed}`.
- Tags: `flow-2x2`, task, world-model variant, policy variant, and seed.
- Training and validation metrics:
  - total loss, prediction MSE, SIGReg loss.
  - flow-matching loss when using a flow world model.
  - action-flow loss when training a flow policy.
  - learning rate, epoch, global step, wall-clock time, and gradient norm when available.
  - validation one-step latent MSE and validation flow loss where applicable.
- Eval metrics:
  - success rate, per-episode successes, evaluation time, seed, task, checkpoint, policy variant, world-model variant.
  - CEM samples/steps/top-k for original policy.
  - flow proposal samples and world-model rollout count for flow policy.
  - videos as W&B media when generated, plus local paths in the manifest.
- Local logging mirrors W&B:
  - `metrics.jsonl` for step/epoch/eval metrics.
  - `run_manifest.yaml` for durable run metadata.
  - `summary.csv` and `summary.md` after aggregating completed runs.

## Run Matrix

Run each cell for seeds `0` and `1`.

| Task | World model | Policy/planner | Variant name |
| --- | --- | --- | --- |
| PushT | original | original CEM | `wm_original_policy_original` |
| PushT | flow | original CEM | `wm_flow_policy_original` |
| PushT | original | flow proposal | `wm_original_policy_flow` |
| PushT | flow | flow proposal | `wm_flow_policy_flow` |
| OGBench-Cube | original | original CEM | `wm_original_policy_original` |
| OGBench-Cube | flow | original CEM | `wm_flow_policy_original` |
| OGBench-Cube | original | flow proposal | `wm_original_policy_flow` |
| OGBench-Cube | flow | flow proposal | `wm_flow_policy_flow` |

## Environment

- A repo-local conda environment is created at `.conda/lewm-flow-2x2`.
- Python version: 3.10.
- Runtime install command:

```bash
.conda/lewm-flow-2x2/bin/python -m pip install "stable-worldmodel[train,env]" wandb hdf5plugin packaging platformdirs psutil pygments pyyaml urllib3 wcwidth
```

- Activation:

```bash
conda activate /storage/project/r-agarg35-0/eliu354/external_repos/le-wm/.conda/lewm-flow-2x2
```

## Validation

- Import smoke:
  - `stable_worldmodel`, `stable_pretraining`, `torch`, `wandb`.
  - repo modules: `jepa`, `module`, flow model/policy modules.
- Shape smoke:
  - One batch through original and flow world models.
  - One action-chunk sample through the flow policy.
- Logging smoke:
  - One tiny train/val run creates a W&B run and local `metrics.jsonl`.
  - One real-environment eval smoke logs success metrics and video paths.
- Formal comparison:
  - Full 2x2 on PushT and Cube, seeds `0` and `1`.
  - Aggregate train, validation, and real-environment eval metrics into `summary.csv` and `summary.md`.

## Assumptions

- OGBench-Cube is the second task because it is a moderately harder manipulation task than PushT.
- Flow v1 uses pure PyTorch flow matching; no extra generative-model dependency is required.
- W&B runs default to online mode. Run `wandb login` before formal jobs; set `WANDB_MODE=offline` only for debugging without network/login.
- Existing untracked user outputs are preserved and not deleted.
