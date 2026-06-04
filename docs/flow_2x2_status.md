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
- Cube dataset smoke:
  - Download/extract job `9427063` completed successfully.
  - Dataset path: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/ogbench/cube_single_expert.h5`.
  - `stable_worldmodel.data.load_dataset()` returned `HDF5Dataset len=1980000`.
  - First sample shapes with `num_steps=4`: `pixels=(4, 3, 224, 224)`, `action=(4, 5)`, `observation=(4, 28)`.

## W&B Status

- W&B login is valid for user `danny010324`.
- Slurm entrypoints now default to `WANDB_MODE=online`. Explicit `WANDB_MODE=offline` can still be passed for local/offline runs.
- Slurm entrypoints now redirect runtime/cache/tmp outputs to project storage under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime`:
  - `XDG_CACHE_HOME`, `XDG_CONFIG_HOME`, `HF_HOME`, `TORCH_HOME`, `MPLCONFIGDIR`, `WANDB_DIR`, `WANDB_CACHE_DIR`, `WANDB_CONFIG_DIR`, and `TMPDIR`.
  - `$HOME` is not changed so W&B can still read the existing `.netrc` credentials, but large runtime artifacts should not be written into the home directory.

## Submitted Jobs

- Cube dataset download/extract:
  - `9427063`, job name `lewm-cube-data`, CPU job, completed with exit code `0:0`.
- First formal submission:
  - PushT world-model training jobs `9426998`, `9426999`, `9427000`, and `9427001` failed after entering W&B offline mode and hitting a `stable_pretraining` offline-run reuse error.
  - Cube world-model training jobs `9427064`, `9427065`, `9427066`, and `9427067` failed because `data=ogb` resolved through HuggingFace instead of the downloaded local HDF5 file.
  - Dependent action-flow/eval jobs `9427119`-`9427144` were canceled after their dependencies became unsatisfiable.
- Resubmitted PushT world-model training:
  - `9428413`, `lewm-pusht-orig-s0`.
  - `9428414`, `lewm-pusht-flow-s0`.
  - `9428415`, `lewm-pusht-orig-s1`.
  - `9428416`, `lewm-pusht-flow-s1`.
- Resubmitted Cube world-model training:
  - `9428417`, `lewm-cube-orig-s0`.
  - `9428418`, `lewm-cube-flow-s0`.
  - `9428419`, `lewm-cube-orig-s1`.
  - `9428420`, `lewm-cube-flow-s1`.
- Resubmitted action-flow and eval jobs:
  - Action-flow jobs: `9428421`, `9428424`, `9428427`, `9428430`, `9428433`, `9428436`, `9428439`, `9428442`.
  - CEM eval jobs: `9428422`, `9428425`, `9428428`, `9428431`, `9428434`, `9428437`, `9428440`, `9428443`.
  - Flow eval jobs: `9428423`, `9428426`, `9428429`, `9428432`, `9428435`, `9428438`, `9428441`, `9428444`.
  - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/resubmit_20260604_151547.tsv`.
- Second formal submission diagnosis:
  - Most world-model jobs exited around 1 hour with `RuntimeError: Pin memory thread exited unexpectedly`.
  - Slurm epilog showed memory usage near the 80GB request, so the likely cause was DataLoader/pinned-memory pressure rather than W&B.
  - Existing partial checkpoints were produced for several variants under each `train/spt/runs/.../checkpoints/last.ckpt`.
  - Old dependency jobs `9428421`-`9428444` and the one still-running training job `9428414` were canceled before resubmission.
- Current 100-epoch resumable submission:
  - World-model jobs request 160GB, use `loader.num_workers=2`, `loader.pin_memory=False`, `loader.persistent_workers=False`, and `loader.prefetch_factor=1`.
  - World-model jobs run with `resume.auto=True`, which resumes from the latest matching `train/spt/runs/**/checkpoints/last.ckpt` when available.
  - PushT world-model training: `9430990`, `9430991`, `9430992`, `9430993`.
  - Cube world-model training: `9430994`, `9430995`, `9430996`, `9430997`.
  - Action-flow jobs: `9430998`, `9431001`, `9431004`, `9431007`, `9431010`, `9431013`, `9431016`, `9431019`.
  - CEM eval jobs: `9430999`, `9431002`, `9431005`, `9431008`, `9431011`, `9431014`, `9431017`, `9431020`.
  - Flow eval jobs: `9431000`, `9431003`, `9431006`, `9431009`, `9431012`, `9431015`, `9431018`, `9431021`.
  - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/resubmit_20260604_173130.tsv`.

Superseded first-submission job IDs:

- PushT world-model training:
  - `9426998`, `lewm-pusht-orig-s0`.
  - `9426999`, `lewm-pusht-flow-s0`.
  - `9427000`, `lewm-pusht-orig-s1`.
  - `9427001`, `lewm-pusht-flow-s1`.
- Cube world-model training, dependency on `9427063` is now satisfied:
  - `9427064`, `lewm-cube-orig-s0`.
  - `9427065`, `lewm-cube-flow-s0`.
  - `9427066`, `lewm-cube-orig-s1`.
  - `9427067`, `lewm-cube-flow-s1`.
- Action-flow and eval jobs were submitted with dependencies:
  - Action-flow jobs: `9427119`, `9427122`, `9427125`, `9427128`, `9427132`, `9427135`, `9427138`, `9427142`.
  - CEM eval jobs: `9427120`, `9427123`, `9427126`, `9427129`, `9427133`, `9427136`, `9427139`, `9427143`.
  - Flow eval jobs: `9427121`, `9427124`, `9427127`, `9427131`, `9427134`, `9427137`, `9427141`, `9427144`.

Latest queue check:

- Current world-model training jobs `9430990`-`9430997` are pending on H200 priority.
- Current action-flow and eval jobs `9430998`-`9431021` are pending on valid dependencies.
- Follow-up check:
  - `9430990`-`9431021` have not started yet; no new Slurm logs or failures were present.
  - World-model jobs request `cpu=8,mem=160G,gres/gpu:h200=1`.
  - No repo-root `wandb/`, `outputs/`, or `multirun/` directories were present after the cleanup.
  - Partial `last.ckpt` files are available for auto-resume for PushT original seed 0/1, PushT flow seed 0/1, and Cube original seed 1.

## Notes

- Generated smoke outputs are intentionally outside git.
- Existing untracked `AGENTS.md` and `reference.md` were left untouched.
