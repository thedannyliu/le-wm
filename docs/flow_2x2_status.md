# Flow 2x2 Status

Date: 2026-06-04

## Implemented

- Branch `exp/flow-2x2-pusht-cube` created.
- `origin` changed to `git@github.com:thedannyliu/le-wm.git`.
- Repo-local conda env created at `.conda/lewm-flow-2x2`.
- Flow world-model predictor, action-flow proposal model, flow proposal solver, W&B/local logging, run manifests, eval logging, and summary aggregation added.
- PushT train data config now uses `pusht_expert_train.h5`, matching the available local HDF5 data.

## Validation

- Clean env import with `PYTHONNOUSERSITE=1`:
  - `stable_worldmodel`, `stable_pretraining`, `torch`, `wandb`, `hdf5plugin`.
- Compile smoke:
  - `jepa.py`, `module.py`, `train.py`, `eval.py`, `experiment_logging.py`, `flow_solver.py`, `train_action_flow.py`, `utils.py`, `summarize_experiments.py`.
- Hydra config smoke:
  - `train.py` with `model=lewm_flow`.
  - `train_action_flow.py` with dummy `world_model`.
  - `eval.py` with `solver=flow`.
- Flow shape smoke:
  - `ConditionalFlowPredictor.flow_loss()` and `ConditionalFlowPredictor.sample()`.
  - `ConditionalActionFlow.flow_loss()` and `ConditionalActionFlow.sample()`.
  - `FlowProposalSolver.solve()` returns `actions` with the expected policy shape.
- Real-environment eval smoke:
  - PushT random policy, `eval.num_eval=1`, `eval.eval_budget=1`.
  - Metrics, manifest, result text, and video were written under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604_smoke/`.
- Train smoke:
  - PushT 1-batch CPU run loaded HDF5 data and wrote local validation metrics.
  - Full CPU fit did not finish within the interactive session; formal training should run on GPU.

## Current Blockers For Formal Runs

- W&B status shows no API key configured. Run `wandb login` or set `WANDB_API_KEY` before formal online W&B jobs.

## Submitted Jobs

- Cube dataset download/extract:
  - `9427063`, job name `lewm-cube-data`, CPU job.
- PushT world-model training:
  - `9426998`, `lewm-pusht-orig-s0`.
  - `9426999`, `lewm-pusht-flow-s0`.
  - `9427000`, `lewm-pusht-orig-s1`.
  - `9427001`, `lewm-pusht-flow-s1`.
- Cube world-model training, all dependent on `afterok:9427063`:
  - `9427064`, `lewm-cube-orig-s0`.
  - `9427065`, `lewm-cube-flow-s0`.
  - `9427066`, `lewm-cube-orig-s1`.
  - `9427067`, `lewm-cube-flow-s1`.
- Action-flow and eval jobs were submitted with dependencies:
  - Action-flow jobs: `9427119`, `9427122`, `9427125`, `9427128`, `9427132`, `9427135`, `9427138`, `9427142`.
  - CEM eval jobs: `9427120`, `9427123`, `9427126`, `9427129`, `9427133`, `9427136`, `9427139`, `9427143`.
  - Flow eval jobs: `9427121`, `9427124`, `9427127`, `9427131`, `9427134`, `9427137`, `9427141`, `9427144`.

Because W&B is not logged in, submitted jobs default to `WANDB_MODE=offline`. Run `wandb sync` after login to upload offline runs.

## Notes

- Generated smoke outputs are intentionally outside git.
- Existing untracked `AGENTS.md` and `reference.md` were left untouched.
