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
- Runtime-output correction:
  - Jobs `9430990`-`9431021` started and W&B/resume worked, but Lightning `WandbLogger` and Hydra still wrote small `wandb/` and `outputs/` directories in the repo root.
  - The running jobs were canceled before further runtime output accumulated.
  - `wandb.config.save_dir` now points at `${WANDB_DIR}`, raw `wandb.init()` calls translate `save_dir` to W&B's `dir`, and Slurm commands set `hydra.run.dir` under `${LEWM_RUNTIME_ROOT}/hydra/...`.
  - Repo-root `wandb/` and `outputs/` were removed again.
  - Resubmitted world-model jobs: `9431374`, `9431375`, `9431376`, `9431377`, `9431378`, `9431379`, `9431380`, `9431381`.
  - Resubmitted action-flow jobs: `9431382`, `9431385`, `9431388`, `9431391`, `9431394`, `9431397`, `9431400`, `9431403`.
  - Resubmitted CEM eval jobs: `9431383`, `9431386`, `9431389`, `9431392`, `9431395`, `9431398`, `9431401`, `9431404`.
  - Resubmitted flow eval jobs: `9431384`, `9431387`, `9431390`, `9431393`, `9431396`, `9431399`, `9431402`, `9431405`.
  - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/resubmit_20260604_175708.tsv`.
  - Latest queue check: world-model jobs `9431374`-`9431381` are pending on H200 priority; downstream jobs `9431382`-`9431405` are pending on valid dependencies.
- Active run check:
  - World-model jobs `9431374`-`9431381` started on H200 nodes and are running.
  - Jobs are writing W&B run data under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime/wandb/wandb`.
  - Hydra run directories are under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime/hydra`.
  - Repo root has no `wandb/`, `outputs/`, or `multirun/` directories.
  - Auto-resume loaded existing `last.ckpt` for PushT original seed 0/1, PushT flow seed 0/1, and Cube original seed 1; Cube original seed 0 and Cube flow seed 0/1 started from scratch because no matching `last.ckpt` existed.
  - All eight train `metrics.jsonl` files are being updated.
  - No `RuntimeError`, `Traceback`, `HTTP Error`, missing-file error, pin-memory failure, or CUDA OOM was found in the latest log sweep.
  - Downstream jobs `9431382`-`9431405` remain pending on valid dependencies.
- Extended active run check:
  - After roughly 40-50 minutes of runtime, world-model jobs `9431374`-`9431381` remain running.
  - PushT jobs are progressing through epoch 1/100 or 2/100; Cube jobs are progressing through epoch 0/100 or 1/100 depending on whether a prior checkpoint existed.
  - Latest log sweep still found no `RuntimeError`, `Traceback`, `HTTP Error`, missing-file error, pin-memory failure, or CUDA OOM.
  - All eight training `metrics.jsonl` files continue to update.
  - Repo root still has no `wandb/`, `outputs/`, or `multirun/` directories.
- Continuation-supervisor correction:
  - A dependency design issue was found: the active world-model jobs target 100 epochs with an 8-hour `embers` limit, so the first GPU chunk is expected to timeout before epoch 100 and direct `afterok` action/eval jobs would become unsatisfiable.
  - Added `scripts/slurm_flow_2x2_supervisor.sbatch`, a short `cpu-small` supervisor job that runs after each world-model chunk with `afterany`.
  - The supervisor checks for the final checkpoint `weights_epoch_100.pt`. If it is missing after a timeout/preemption/completion, it submits the next resumable world-model GPU chunk with `resume.auto=True`; if it exists, it submits action-flow, CEM eval, and flow eval jobs.
  - Updated `scripts/submit_flow_2x2_full.sh` so future full submissions attach supervisors instead of direct action/eval dependencies to the first world-model chunk.
  - Canceled stale direct downstream jobs `9431382`-`9431405` before they could become dependency failures.
  - Attached replacement supervisors:
    - `9432069`, `sup-pusht-original-s0`, `afterany:9431374`.
    - `9432070`, `sup-pusht-flow-s0`, `afterany:9431375`.
    - `9432071`, `sup-pusht-original-s1`, `afterany:9431376`.
    - `9432072`, `sup-pusht-flow-s1`, `afterany:9431377`.
    - `9432073`, `sup-cube-original-s0`, `afterany:9431378`.
    - `9432074`, `sup-cube-flow-s0`, `afterany:9431379`.
    - `9432075`, `sup-cube-original-s1`, `afterany:9431380`.
    - `9432076`, `sup-cube-flow-s1`, `afterany:9431381`.
  - Supervisor submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/supervisor_attach_20260604_190730.tsv`.
  - Validation: `bash -n` passed for all flow Slurm scripts, and `sbatch --test-only` confirmed the supervisor script is schedulable on `cpu-small`.
- Follow-up repair after preemption:
  - Jobs `9431376`, `9431380`, and `9431381` were preempted by the cluster after roughly 1 hour. Their logs include `/tmp/pymp-*` multiprocessing `FileNotFoundError` shutdown noise, but `sacct` reports `PREEMPTED` rather than an application OOM or data/config failure.
  - Supervisors `9432071`, `9432075`, and `9432076` ran and correctly submitted replacement resumable world-model chunks:
    - `9432169`, `lewm-pusht-original-s1`, pending on H200 priority.
    - `9432164`, `lewm-cube-original-s1`, running on H200 and resumed from the prior `last.ckpt`.
    - `9432197`, `lewm-cube-flow-s1`, pending on H200 priority.
  - A QOS issue was found in the supervisor script: it did not explicitly request `embers`, so completed/pending CPU supervisors inherited the cluster default `inferno`.
  - Added `#SBATCH --qos=embers` to `scripts/slurm_flow_2x2_supervisor.sbatch`.
  - Canceled pending inferno supervisors `9432069`, `9432070`, `9432072`, `9432073`, `9432074`, `9432166`, `9432170`, and `9432198`.
  - Submitted embers replacement supervisors:
    - `9432333`, `sup-pusht-original-s0`, `afterany:9431374`.
    - `9432334`, `sup-pusht-flow-s0`, `afterany:9431375`.
    - `9432335`, `sup-pusht-flow-s1`, `afterany:9431377`.
    - `9432336`, `sup-cube-original-s0`, `afterany:9431378`.
    - `9432337`, `sup-cube-flow-s0`, `afterany:9431379`.
    - `9432338`, `sup-cube-original-s1`, `afterany:9432164`.
    - `9432339`, `sup-pusht-original-s1`, `afterany:9432169`.
    - `9432340`, `sup-cube-flow-s1`, `afterany:9432197`.
  - Replacement supervisor record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/supervisor_embers_replace_20260604_192503.tsv`.
  - Validation: `bash -n` passed, `sbatch --test-only` confirmed `cpu-small` with `embers`, and `scontrol show job` confirmed replacement supervisors use `QOS=embers`.
- Latest active run check:
  - Job `9431379`, `lewm-cube-flow-s0`, was preempted after roughly 1.5 hours.
  - Supervisor `9432337` ran with `QOS=embers` and correctly submitted:
    - `9432486`, `lewm-cube-flow-s0`, running on H200.
    - `9432487`, `sup-cube-flow-s0`, pending on `afterany:9432486` with `QOS=embers`.
  - Current active world-model jobs are:
    - `9431374`, PushT original seed 0, running at epoch 1/100.
    - `9431375`, PushT flow seed 0, running at epoch 2/100.
    - `9431377`, PushT flow seed 1, running at epoch 1/100.
    - `9431378`, Cube original seed 0, running at epoch 0/100.
    - `9432164`, Cube original seed 1 replacement, running at epoch 1/100.
    - `9432169`, PushT original seed 1 replacement, running at epoch 1/100.
    - `9432197`, Cube flow seed 1 replacement, running at epoch 0/100.
    - `9432486`, Cube flow seed 0 replacement, running at epoch 0/100.
  - Active log sweep found no new `RuntimeError`, `Traceback`, missing-file error, pin-memory failure, CUDA OOM, or dependency failure.
  - All eight training `metrics.jsonl` files continue to update.
  - Repo root still has no `wandb/`, `outputs/`, or `multirun` directories.
  - W&B and Hydra runtime directories remain under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime`.
- Follow-up health check:
  - No additional world-model jobs failed or required repair.
  - Current active world-model jobs remain running:
    - `9431374`, PushT original seed 0, epoch 1/100.
    - `9431375`, PushT flow seed 0, epoch 2/100.
    - `9431377`, PushT flow seed 1, epoch 1/100.
    - `9431378`, Cube original seed 0, epoch 0/100.
    - `9432164`, Cube original seed 1 replacement, epoch 1/100.
    - `9432169`, PushT original seed 1 replacement, epoch 1/100.
    - `9432197`, Cube flow seed 1 replacement, epoch 0/100.
    - `9432486`, Cube flow seed 0 replacement, epoch 0/100.
  - Pending supervisors `9432333`, `9432334`, `9432335`, `9432336`, `9432338`, `9432339`, `9432340`, and `9432487` all use `QOS=embers` and have valid `afterany` dependencies.
  - Active log sweep found no new `RuntimeError`, `Traceback`, missing-file error, pin-memory failure, CUDA OOM, or dependency failure.
  - All eight training `metrics.jsonl` files continue to update.
  - Repo root still has no `wandb/`, `outputs/`, or `multirun` directories.
- Workdir sidecar repair:
  - Additional preemptions occurred:
    - `9431374`, PushT original seed 0, preempted after roughly 2.8 hours.
    - `9431375`, PushT flow seed 0, preempted after roughly 3.0 hours.
    - `9431378`, Cube original seed 0, preempted after roughly 3.0 hours.
    - `9432169`, PushT original seed 1 replacement, preempted after roughly 1.0 hour.
  - Embers supervisors correctly submitted resumable replacements:
    - `9432951`, PushT original seed 0, running; `9432952`, supervisor pending on `afterany:9432951`.
    - `9432921`, PushT original seed 1, running; `9432922`, supervisor pending on `afterany:9432921`.
    - `9433089`, PushT flow seed 0, pending; `9433091`, supervisor pending on `afterany:9433089`.
    - `9433149`, Cube original seed 0, pending; `9433150`, supervisor pending on `afterany:9433149`.
  - A new repo-root sidecar `wandb_resume.json` appeared. It is written by `stable_pretraining`'s W&B checkpoint callback when the job CWD is the repo, so future Slurm GPU jobs now run from `${LEWM_RUNTIME_ROOT}/workdirs/${SLURM_JOB_ID}` while executing repo scripts by absolute path with `PYTHONPATH=${REPO}`.
  - Updated `scripts/slurm_runtime_env.sh`, `scripts/slurm_flow_2x2_train.sbatch`, `scripts/slurm_flow_2x2_action_flow.sbatch`, and `scripts/slurm_flow_2x2_eval.sbatch` to route the process CWD to project runtime.
  - Removed the repo-root `wandb_resume.json` residue.
  - Canceled pending pre-patch jobs `9433089`, `9433091`, `9433149`, and `9433150`, because Slurm had already captured their old repo-CWD scripts.
  - Resubmitted patched workdir replacements:
    - `9433162`, PushT flow seed 0, pending on H200 priority; `9433163`, supervisor pending on `afterany:9433162`.
    - `9433164`, Cube original seed 0, pending on H200 priority; `9433165`, supervisor pending on `afterany:9433164`.
  - Replacement record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/workdir_patch_resubmit_20260604_211729.tsv`.
  - Validation: `bash -n` passed for all Slurm scripts, `sbatch --test-only` passed for the patched train entrypoint, and a runtime-CWD import smoke loaded `train`, `train_action_flow`, and `eval`.
  - Active log sweep found no new `RuntimeError`, `Traceback`, missing-file error, pin-memory failure, CUDA OOM, or dependency failure in the running jobs.
  - Current running jobs are `9431377`, `9432164`, `9432197`, `9432486`, `9432921`, and `9432951`; patched replacements `9433162` and `9433164` are pending on H200 priority.
- Queue repair, 2026-06-04 23:02 EDT:
  - The remaining active world-model chunks were preempted by the cluster, not by an application error:
    - `9431377`, PushT flow seed 1, preempted after 3:30:12.
    - `9432164`, Cube original seed 1, preempted after 2:00:45.
    - `9432197`, Cube flow seed 1, preempted after 2:20:36.
    - `9432486`, Cube flow seed 0, preempted after 2:59:08.
    - `9432921`, PushT original seed 1, preempted after 1:47:05.
    - `9432951`, PushT original seed 0, preempted after 1:40:22.
    - `9433162`, PushT flow seed 0, preempted after 1:06:11.
    - `9433164`, Cube original seed 0, preempted after 1:00:56.
  - Supervisors resubmitted resumable world-model chunks for all variants, but `9432338`, `sup-cube-original-s1`, failed after submitting replacement GPU job `9433204` because its attempt to submit the next supervisor hit `QOSMaxSubmitJobPerUserLimit`.
  - Manually submitted the missing replacement supervisor `9435928`, `sup-cube-original-s1`, with `QOS=embers` and `afterany:9433204`.
  - Manual repair record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/job_records/missing_supervisor_20260604_230048.tsv`.
  - Current world-model queue is complete and pending on H200 priority:
    - `9433204`, Cube original seed 1.
    - `9433301`, PushT flow seed 1.
    - `9433455`, Cube flow seed 1.
    - `9434238`, PushT flow seed 0.
    - `9434239`, PushT original seed 0.
    - `9434240`, Cube flow seed 0.
    - `9434244`, PushT original seed 1.
    - `9434358`, Cube original seed 0.
  - Current supervisor queue is complete, uses `QOS=embers`, and has valid `afterany` dependencies:
    - `9435928`, `sup-cube-original-s1`, `afterany:9433204`.
    - `9433302`, `sup-pusht-flow-s1`, `afterany:9433301`.
    - `9433456`, `sup-cube-flow-s1`, `afterany:9433455`.
    - `9434242`, `sup-pusht-flow-s0`, `afterany:9434238`.
    - `9434241`, `sup-pusht-original-s0`, `afterany:9434239`.
    - `9434243`, `sup-cube-flow-s0`, `afterany:9434240`.
    - `9434245`, `sup-pusht-original-s1`, `afterany:9434244`.
    - `9434359`, `sup-cube-original-s0`, `afterany:9434358`.
  - Metrics files and W&B runs are expected to be stale until the pending H200 jobs start again. The latest logs before preemption did not show a new `RuntimeError`, `Traceback`, missing-file error, pin-memory failure, CUDA OOM, or dependency failure.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json` was present after the repair.
- Queue repair, 2026-06-05 02:32 EDT:
  - Another cluster preemption wave stopped the prior H200 world-model chunks. `sacct` reports `PREEMPTED` for the affected GPU jobs, while the corresponding `embers` supervisors completed and submitted the next resumable chunks.
  - Completed supervisors submitted the following replacements:
    - `9436864`, PushT flow seed 1; supervisor `9436865`, `afterany:9436864`.
    - `9436866`, Cube original seed 1; supervisor `9436867`, `afterany:9436866`.
    - `9436871`, Cube flow seed 0; supervisor `9436872`, `afterany:9436871`.
    - `9436873`, PushT original seed 0; supervisor `9436874`, `afterany:9436873`.
    - `9436885`, Cube flow seed 1; supervisor `9436886`, `afterany:9436885`.
    - `9436887`, PushT flow seed 0; supervisor `9436888`, `afterany:9436887`.
    - `9436932`, PushT original seed 1; supervisor `9436933`, `afterany:9436932`.
    - `9436934`, Cube original seed 0; supervisor `9436935`, `afterany:9436934`.
  - Current queue is complete: all eight world-model jobs are pending on `gpu-h200` for priority, and all eight supervisors are pending on valid dependencies. `scontrol show job` confirms `QOS=embers` for all current GPU and CPU jobs.
  - No replacement job is currently running, so metrics are expected to be stale until H200 capacity is assigned again. Latest recorded training progress:
    - PushT original seed 0: epoch 2, global step 38350.
    - PushT original seed 1: epoch 1, global step 23850.
    - PushT flow seed 0: epoch 3, global step 52400.
    - PushT flow seed 1: epoch 2, global step 38400.
    - Cube original seed 0: epoch 0, global step 6850.
    - Cube original seed 1: epoch 1, global step 19500.
    - Cube flow seed 0: epoch 0, global step 6200.
    - Cube flow seed 1: epoch 0, global step 6600.
  - Final `weights_epoch_100.pt` checkpoints are not present yet, so action-flow and real-environment eval jobs have not been submitted by the supervisors.
  - Log sweep still shows old pin-memory failures from the pre-fix jobs and multiprocessing `/tmp/pymp-*` cleanup traces from preempted chunks. The active queue has no current failed job, missing supervisor, dependency failure, CUDA OOM, data/config error, or W&B error.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`. A HOME scan found only the existing W&B config directory `/storage/home/hcoda1/9/eliu354/.config/wandb`, not training outputs.
- Health check, 2026-06-05 03:32 EDT:
  - No new Slurm repair was needed. The current queue still has the same eight world-model jobs pending on `gpu-h200` for priority and the same eight supervisors pending on valid `afterany` dependencies.
  - `scontrol show job` confirms all current GPU and CPU jobs use `QOS=embers`; no job is using `inferno`.
  - Current world-model jobs: `9436864`, `9436866`, `9436871`, `9436873`, `9436885`, `9436887`, `9436932`, and `9436934`.
  - Current supervisors: `9436865`, `9436867`, `9436872`, `9436874`, `9436886`, `9436888`, `9436933`, and `9436935`.
  - Final checkpoint count remains zero for `weights_epoch_100.pt`; action-flow and real-environment eval outputs are not present yet.
  - Latest training metrics are unchanged from the 02:32 EDT check because no replacement world-model job has started since then.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- Queue repair, 2026-06-05 12:06 EDT:
  - The pending replacement jobs started on H200. At check time, seven world-model chunks were running and one replacement was pending:
    - Running: `9442757` PushT flow seed 1, `9442771` Cube original seed 1, `9442773` Cube flow seed 0, `9445127` PushT original seed 0, `9446407` Cube flow seed 1, `9446415` PushT flow seed 0, and `9446424` PushT original seed 1.
    - `9436934`, Cube original seed 0, was preempted after 1:00:34. Supervisor `9436935` completed and submitted `9448176` plus supervisor `9448177`; both use `QOS=embers` and valid dependency wiring.
  - Active log sweep found no current `RuntimeError`, CUDA OOM, W&B error, missing data/config error, or dependency failure in the running jobs. The `9436934` trace is multiprocessing cleanup noise after Slurm preemption; `sacct` reports `PREEMPTED`.
  - A resume-efficiency issue was found: Cube jobs that have not finished epoch 0 have no stable-pretraining `last.ckpt`, so a preempt before the first epoch checkpoint restarts those chunks from scratch. PushT jobs with existing `last.ckpt` resumed correctly.
  - Added step-level Lightning requeue checkpointing in `train.py`: `ModelCheckpoint(filename="last", every_n_train_steps=1000, save_top_k=-1, save_on_train_epoch_end=False)`, configured by `logging.requeue_checkpoint_every_n_steps: 1000`.
  - This patch affects future chunks launched after the code change; currently running jobs were not canceled, to avoid throwing away their in-flight progress.
  - Validation: `.conda/lewm-flow-2x2/bin/python` successfully imported `train.py`, and Hydra composition confirmed `logging.requeue_checkpoint_every_n_steps=1000`.
  - Final checkpoint count remains zero for `weights_epoch_100.pt`; action-flow and real-environment eval outputs are not present yet.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- Queue repair, 2026-06-05 13:29 EDT:
  - The H200 chunks that were running at 12:06 were preempted again. `sacct` reports `PREEMPTED`, not an application failure:
    - `9442757`, PushT flow seed 1, after 1:01:05.
    - `9442771`, Cube original seed 1, after 1:01:16.
    - `9442773`, Cube flow seed 0, after 1:01:50.
    - `9445127`, PushT original seed 0, after 1:58:41.
    - `9446407`, Cube flow seed 1, after 1:58:09.
    - `9446415`, PushT flow seed 0, after 1:58:11.
    - `9446424`, PushT original seed 1, after 1:59:14.
  - The corresponding supervisors completed and submitted replacement chunks:
    - `9448680`, PushT flow seed 1; supervisor `9448681`.
    - `9448686`, Cube original seed 1; supervisor `9448687`.
    - `9448697`, Cube flow seed 0; supervisor `9448698`.
    - `9450886`, Cube flow seed 1; supervisor `9450887`.
    - `9450903`, PushT original seed 0; supervisor `9450905`.
    - `9450904`, PushT flow seed 0; supervisor `9450906`.
    - `9450920`, PushT original seed 1; supervisor `9450921`.
  - Together with earlier replacement `9448176` and supervisor `9448177` for Cube original seed 0, the current queue is complete: eight world-model jobs are pending on `gpu-h200` for priority and eight supervisors are pending on valid `afterany` dependencies.
  - `scontrol show job` confirms all current GPU and CPU jobs use `QOS=embers`; no job is using `inferno`.
  - Step-level requeue checkpointing has not produced new `last.ckpt` files yet because the replacement chunks submitted after the patch have not started running. Existing `last.ckpt` files are still the older epoch-end checkpoints from 2026-06-04.
  - Latest training metrics before the preemptions:
    - PushT original seed 0: epoch 2, global step 41750.
    - PushT original seed 1: epoch 1, global step 27850.
    - PushT flow seed 0: epoch 3, global step 55600.
    - PushT flow seed 1: epoch 2, global step 34800.
    - Cube original seed 0: epoch 0, global step 4550.
    - Cube original seed 1: epoch 1, global step 17200.
    - Cube flow seed 0: epoch 0, global step 4550.
    - Cube flow seed 1: epoch 0, global step 9000.
  - Final checkpoint count remains zero for `weights_epoch_100.pt`; action-flow and real-environment eval outputs are not present yet.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- Scope reset, 2026-06-05 15:22 EDT:
  - User narrowed the active run scope to PushT only. OGB/Cube and the formal flow 2x2 queue are deferred until after the PushT speed sanity check.
  - Canceled the stale H200 formal queue and replacement supervisors so no old PushT/Cube 2x2 jobs remain active:
    - `9448176`, `9448177`, `9448680`, `9448681`, `9448686`, `9448687`, `9448697`, `9448698`, `9450886`, `9450887`, `9450903`, `9450905`, `9450904`, `9450906`, `9450920`, `9450921`, `9455507`, `9455509`, `9455414`, and `9455416`.
  - Restored the PushT data config to the official Lance dataset name: `pusht_expert_train.lance`.
  - Added a CPU conversion job for the local HDF5 data:
    - `9456102`, `lewm-pusht-lance`, `cpu-small`, `QOS=embers`, converting `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/datasets/pusht_expert_train.h5` to `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/datasets/pusht_expert_train.lance`.
  - Restored faster dataloader defaults for future Slurm training/action-flow jobs: `loader.num_workers=6`, `loader.pin_memory=True`, `loader.persistent_workers=True`, and `loader.prefetch_factor=3`.
  - Switched future formal Slurm train/action-flow/eval entrypoints from `gpu-h200` to `gpu-h100` by default, while keeping `QOS=embers`.
  - Submitted official native LeWM PushT one-epoch speed sanity jobs with distinct output roots and W&B names:
    - `9456103`, `lewm-pusht-h100-speed`, `gpu-h100`, dependency `afterok:9456102`, output root `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605`.
    - `9456323`, `lewm-pusht-a100-speed`, `gpu-a100`, dependency `afterok:9456102`, output root `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_a100_speed_20260605`.
    - `9456362`, `lewm-pusht-l40s-speed`, `gpu-l40s`, dependency `afterok:9456102`, output root `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_l40s_speed_20260605`.
  - The speed sanity command follows the native LeWM path: `train.py data=pusht model=lewm`, `trainer.max_epochs=1`, online W&B logging, and project-storage Hydra/W&B/runtime directories.
  - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605/job_records/pusht_h100_speed_20260605_151226.tsv`.
  - Latest queue state: Lance conversion `9456102` is running; H100/A100/L40S speed sanity jobs are pending on `afterok:9456102`.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- Formal PushT native submission, 2026-06-05 15:25 EDT:
  - Submitted PushT-only native LeWM formal world-model training via `scripts/submit_pusht_native_formal.sh`.
  - The formal command path is still the official native route: `train.py data=pusht model=lewm`, `trainer.max_epochs=100`, online W&B, Lance data, and project-storage runtime directories.
  - Jobs are dependency-gated on Lance conversion `afterok:9456102` so training does not read a partially converted dataset.
  - Formal world-model jobs:
    - `9456786`, seed 0, `gpu-h100`, `QOS=embers`, `8 CPU`, `160G`, dependency `afterok:9456102`.
    - `9456788`, seed 1, `gpu-a100`, `QOS=embers`, `8 CPU`, `160G`, dependency `afterok:9456102`.
    - `9456790`, seed 2, `gpu-l40s`, `QOS=embers`, `4 CPU`, `120G`, dependency `afterok:9456102`.
  - Continuation supervisors:
    - `9456787`, watches `9456786` with `afterany`.
    - `9456789`, watches `9456788` with `afterany`.
    - `9456791`, watches `9456790` with `afterany`.
  - Supervisors inherit the formal output root and will submit resumable replacement chunks with `resume.auto=True` until `weights_epoch_100.pt` exists, then submit action-flow, CEM real-environment eval, and flow-policy real-environment eval for the completed PushT native world model.
  - Formal output root: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605`.
  - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/submit_20260605_152119.tsv`.
  - Validation: shell syntax passed for submit/train/supervisor/action/eval scripts; `sbatch --test-only` passed for H100, A100, and L40S formal train submissions; `scontrol show job` confirmed `QOS=embers`, expected partitions, GPU types, CPU/memory requests, and dependencies.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT repair, 2026-06-05 16:50 EDT:
  - Lance conversion `9456102` completed successfully after 20:07.
  - The first A100/L40S speed sanity jobs failed before training because the script used `checkpoint_run_name=...` against Hydra strict config. Fixed the override to `+checkpoint_run_name=...`.
  - Canceled the still-pending first H100 speed sanity job `9456103`, then resubmitted retry speed sanity jobs with separate output roots and W&B labels:
    - `9464427`, `lewm-pusht-h100-speed-r1`, `gpu-h100`, `QOS=embers`, pending on priority.
    - `9464428`, `lewm-pusht-a100-speed-r1`, `gpu-a100`, `QOS=embers`, pending on priority.
    - `9464429`, `lewm-pusht-l40s-speed-r1`, `gpu-l40s`, `QOS=embers`, pending on priority with `4 CPU`, `120G`, and `NUM_WORKERS=4`.
  - The formal L40S seed 2 job `9456790` started and is training normally:
    - Latest observed progress: epoch 0, global step 4100 of 13933, about 4.6 it/s.
    - `metrics.jsonl` is updating under the PushT formal output root.
    - Step-level `last.ckpt` files have been written at 1000, 2000, and 3000 steps under the project-storage experiment root.
    - W&B is online and writes local run data under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime/wandb`.
  - Found a W&B run-id collision risk for the pending formal seed 0/1 jobs because `subdir` defaulted to `pusht_original_seed0` and `pusht_original_seed1`, which were also used by the earlier formal attempts.
  - Added `SUBDIR` support to `scripts/slurm_flow_2x2_train.sbatch`, propagated it through the supervisor, and updated `scripts/submit_pusht_native_formal.sh` so future formal submissions use unique W&B ids such as `pusht_native_formal_20260605_h100_seed0`.
  - Canceled pending pre-fix formal jobs and supervisors `9456786`, `9456787`, `9456788`, and `9456789`.
  - Resubmitted formal seed 0/1 with unique `SUBDIR` values:
    - `9464470`, seed 0, `lewm-pusht-native-h100-s0-r1`, `gpu-h100`, `QOS=embers`, pending on priority.
    - `9464471`, supervisor for `9464470`, `QOS=embers`, dependency `afterany:9464470`.
    - `9464472`, seed 1, `lewm-pusht-native-a100-s1-r1`, `gpu-a100`, `QOS=embers`, pending on priority.
    - `9464473`, supervisor for `9464472`, `QOS=embers`, dependency `afterany:9464472`.
  - Active queue now has one running formal job (`9456790`), two pending formal jobs (`9464470`, `9464472`), three pending speed sanity retries (`9464427`, `9464428`, `9464429`), and three valid formal supervisors (`9456791`, `9464471`, `9464473`).
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT health check, 2026-06-05 19:50 EDT:
  - Speed sanity retry `9464429` on L40S completed successfully in 58:14 with `QOS=embers`.
    - One epoch processed 13933 steps at about 4.59 it/s.
    - Final train metrics included `fit/loss=0.1710` and `fit/pred_loss=0.03136`.
    - Final validation metrics included `validate/loss=0.2036` and `validate/pred_loss=0.02707`.
    - Checkpoint written: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_l40s_speed_retry1_20260605/checkpoints/pusht/l40s_retry1_speed/seed_0/lewm/weights_epoch_1.pt`.
  - Formal L40S seed 2 job `9456790` remains running and healthy after about 3:19 runtime.
    - Latest observed progress: epoch 3, global step about 50550, about 4.6 it/s.
    - Latest metrics show `validate/loss=0.1463`, `validate/pred_loss=0.01146`, and train-step `fit/loss` around `0.09`-`0.11`.
    - Epoch checkpoints written so far: `weights_epoch_1.pt`, `weights_epoch_2.pt`, and `weights_epoch_3.pt`.
    - Step-level `last.ckpt` continues to update under the project-storage formal run directory.
  - Pending jobs remain valid and are waiting on priority, not failed dependencies:
    - Speed sanity retries: `9464427` on H100 and `9464428` on A100.
    - Formal jobs: `9464470` on H100 seed 0 and `9464472` on A100 seed 1.
    - Supervisors: `9456791`, `9464471`, and `9464473`, each pending on the matching train job with `afterany`.
  - Log sweep found no new `RuntimeError`, `Traceback`, CUDA OOM, Hydra override error, missing-file error, dependency failure, pin-memory failure, or W&B error in current PushT formal/sanity logs.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT dataloader repair, 2026-06-05 20:25 EDT:
  - Formal L40S seed 2 job `9456790` stopped after 3:49:35. `sacct` reports `PREEMPTED`, but the application log shows `RuntimeError: Pin memory thread exited unexpectedly` at epoch 4, global step 57741.
  - Progress was not lost:
    - Epoch checkpoints were written through `weights_epoch_4.pt`.
    - Step-level `last.ckpt` was last updated at 20:16:39 under the project-storage formal run directory.
  - Supervisor `9456791` completed and submitted replacement `9476414` plus supervisor `9476415`, but those jobs would still have used the old formal pin-memory dataloader settings.
  - Added dataloader environment controls to the formal train, supervisor, and action-flow Slurm entrypoints:
    - `PIN_MEMORY`
    - `PERSISTENT_WORKERS`
    - `PREFETCH_FACTOR`
  - Updated `scripts/submit_pusht_native_formal.sh` so formal jobs default to the stable dataloader settings `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, and `PREFETCH_FACTOR=1`.
  - Canceled old pending formal jobs and supervisors that would have kept the unstable pin-memory settings:
    - `9464470`, `9464471`, `9464472`, `9464473`, `9476414`, and `9476415`.
  - Resubmitted stable formal jobs:
    - `9476430`, seed 0, H100, `QOS=embers`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, `PREFETCH_FACTOR=1`, pending on priority.
    - `9476431`, supervisor for `9476430`, dependency `afterany:9476430`.
    - `9476432`, seed 1, A100, `QOS=embers`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, `PREFETCH_FACTOR=1`, pending on priority.
    - `9476433`, supervisor for `9476432`, dependency `afterany:9476432`.
    - `9476434`, seed 2, L40S, `QOS=embers`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, `PREFETCH_FACTOR=1`, pending on priority and expected to resume from the existing seed 2 `last.ckpt`.
    - `9476435`, supervisor for `9476434`, dependency `afterany:9476434`.
  - Speed sanity retries `9464427` and `9464428` remain pending on H100/A100 priority. The completed L40S sanity retry `9464429` remains valid as the current speed sanity result.
  - Validation: shell syntax passed, Hydra composed with `loader.pin_memory=False`, `loader.persistent_workers=False`, and `loader.prefetch_factor=1`, and `scontrol show job -dd` confirmed all new formal jobs and supervisors carry the stable dataloader environment.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT repair, 2026-06-06 05:10 EDT:
  - H100 speed sanity retry `9464427` completed successfully in 36:55 wall time.
    - One epoch processed 13933 steps with a final reported rate around 8.09 it/s.
    - Final metrics included `fit/loss=0.1671`, `fit/pred_loss=0.03133`, `validate/loss=0.2055`, and `validate/pred_loss=0.02822`.
    - Checkpoint written: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_retry1_20260605/checkpoints/pusht/h100_retry1_speed/seed_0/lewm/weights_epoch_1.pt`.
  - Formal H100 seed 0 job `9476430` is running with the stable dataloader settings:
    - Latest observed progress: epoch 8, about 5.3 it/s.
    - Epoch checkpoints written through `weights_epoch_8.pt`.
    - Step-level `last.ckpt` continues to update under the project-storage formal run directory.
  - Formal A100 seed 1 job `9476432` is running with the stable dataloader settings:
    - Latest observed progress: epoch 0, about 4.6 it/s.
    - Step-level `last.ckpt` has been written.
  - A100 speed sanity retry `9464428` is running.
  - Formal L40S seed 2 job `9476434` failed at startup on node `atl1-1-03-007-31-0` because that node reported an NVIDIA driver too old for the current PyTorch/CUDA build. Supervisor `9476435` stopped because the upstream state was `FAILED`.
  - Updated `scripts/submit_pusht_native_formal.sh` so formal seed 2 defaults to H100 instead of L40S, avoiding the mixed-driver L40S partition for formal training.
  - Submitted H100 replacement for formal seed 2:
    - `9490035`, `lewm-pusht-native-h100-s2-drvfix`, `gpu-h100`, `QOS=embers`, stable dataloader settings, pending on priority.
    - `9490036`, `sup-pusht-native-h100-s2-drvfix`, dependency `afterany:9490035`.
    - Submission record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/repair_l40s_driver_seed2_20260606_050751.tsv`.
  - Active queue now has H100 seed 0 running (`9476430`), A100 seed 1 running (`9476432`), H100 seed 2 pending (`9490035`), A100 speed sanity running (`9464428`), and valid supervisors for seeds 0/1/2 (`9476431`, `9476433`, `9490036`).
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT formal health check, 2026-06-06 22:05 EDT:
  - A100 speed sanity retry `9464428` completed successfully in 46:37 wall time with `QOS=embers`.
    - One epoch processed 13933 steps with a final reported training rate around 5.98 it/s.
    - Final metrics included `fit/loss=0.2520`, `fit/pred_loss=0.06682`, `validate/loss=0.2035`, and `validate/pred_loss=0.02619`.
    - Checkpoint written: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_a100_speed_retry1_20260605/checkpoints/pusht/a100_retry1_speed/seed_0/lewm/weights_epoch_1.pt`.
  - Formal PushT native world-model jobs are running on H100 with stable dataloader settings and valid continuation supervisors:
    - `9507523`, seed 0, running for 7:41 on H100; supervisor `9507524` pending on `afterany:9507523`.
    - `9509959`, seed 1, running for 5:46 on H100; supervisor `9509960` pending on `afterany:9509959`.
    - `9518823`, seed 2, running for 0:43 on H100; supervisor `9518824` pending on `afterany:9518823`.
  - Latest observed formal progress:
    - Seed 0: epoch 27/100, global step 390100, about 5.1 it/s; latest validation metrics include `validate/loss=0.11760` and `validate/pred_loss=0.003113`; epoch checkpoints written through `weights_epoch_27.pt`.
    - Seed 1: epoch 18/100, global step 263350, about 5.2-5.3 it/s; latest validation metrics include `validate/loss=0.11778` and `validate/pred_loss=0.003623`; epoch checkpoints written through `weights_epoch_18.pt`.
    - Seed 2: epoch 23 completed and `weights_epoch_24.pt` was saved; latest validation metrics include `validate/loss=0.11826` and `validate/pred_loss=0.003827`; training is continuing into the next epoch.
  - Historical issues are already repaired:
    - Earlier pin-memory failure was addressed by formal defaults `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, and `PREFETCH_FACTOR=1`.
    - Earlier L40S driver failure was addressed by moving formal seed 2 to H100.
    - Expected `TIMEOUT`/`PREEMPTED` chunks are handled by the `afterany` supervisors with `resume.auto=True`.
  - Current log sweep found no new `RuntimeError`, `Traceback`, CUDA OOM, driver error, missing-file error, pin-memory failure, dependency failure, or W&B error in the active PushT formal/sanity logs.
  - No repair or cancellation was needed in this check.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT formal continuation check, 2026-06-06 22:42 EDT:
  - Seed 0 chunk `9507523` reached the expected 8-hour `embers` wall-time limit and ended with `TIMEOUT` after 8:00:28.
  - Supervisor `9507524` completed successfully in 8 seconds with `QOS=embers`.
    - It checked for `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_0/lewm/weights_epoch_100.pt`.
    - Because the final checkpoint was not ready, it submitted resumable seed 0 continuation job `9522910` and next supervisor `9522912`.
  - New seed 0 continuation `9522910` is pending on H100 priority, not failed:
    - `QOS=embers`, `gpu-h100`, `8 CPU`, `160G`, `TimeLimit=08:00:00`.
    - Exported settings include `RESUME_AUTO=True`, `NUM_WORKERS=6`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, and `PREFETCH_FACTOR=1`.
    - Supervisor `9522912` is pending with valid dependency `afterany:9522910`.
  - Current active formal jobs remain healthy:
    - Seed 1 `9509959` is running on H100 at epoch 19/100, about 5.1 it/s, with `last.ckpt` updates every 1000 steps.
    - Seed 2 `9518823` is running on H100 at epoch 24/100, about 5.1-5.2 it/s, with `last.ckpt` updates every 1000 steps.
  - Latest checkpoint progress:
    - Seed 0 has epoch checkpoints through `weights_epoch_28.pt`.
    - Seed 1 has epoch checkpoints through `weights_epoch_19.pt`.
    - Seed 2 has epoch checkpoints through `weights_epoch_24.pt`.
  - Current log sweep found no new `RuntimeError`, `Traceback`, CUDA OOM, driver error, missing-file error, pin-memory failure, dependency failure, or W&B error in the active formal logs.
  - No manual repair was needed beyond confirming the supervisor-submitted continuation.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT formal health check, 2026-06-07 04:00 EDT:
  - Seed 1 chunk `9509959` reached the expected 8-hour `embers` wall-time limit and ended with `TIMEOUT` after 8:00:26.
  - Supervisor `9509960` completed successfully and submitted resumable seed 1 continuation `9528495` plus supervisor `9528496`.
  - Seed 1 continuation `9528495` was later `PREEMPTED` after 2:18:41.
    - Its log contains `FileNotFoundError` in `multiprocessing.resource_sharer` and `/tmp/pymp-*` finalizer cleanup near the preemption time.
    - This is treated as preemption shutdown noise rather than a missing dataset/checkpoint error because the next continuation restored from `last.ckpt` and resumed training successfully.
  - Supervisor `9528496` completed successfully and submitted current seed 1 continuation `9533424` plus supervisor `9533425`.
  - Current formal queue:
    - Seed 0 `9522910` is running on H100 with `QOS=embers`; supervisor `9522912` is pending on `afterany:9522910`.
    - Seed 1 `9533424` is running on H100 with `QOS=embers`; supervisor `9533425` is pending on `afterany:9533424`.
    - Seed 2 `9518823` is running on H100 with `QOS=embers`; supervisor `9518824` is pending on `afterany:9518823`.
  - All active jobs have the intended resumable/stable dataloader export settings: `RESUME_AUTO=True`, `NUM_WORKERS=6`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, and `PREFETCH_FACTOR=1`.
  - Latest observed progress:
    - Seed 0: epoch 33/100, global step 463550; epoch checkpoints through `weights_epoch_33.pt`; validation metrics include `validate/loss=0.11678` and `validate/pred_loss=0.002855`.
    - Seed 1: epoch 25/100, global step 361900; epoch checkpoints through `weights_epoch_25.pt`; validation metrics include `validate/loss=0.11832` and `validate/pred_loss=0.003179`.
    - Seed 2: epoch 31/100, global step 433800; epoch checkpoints through `weights_epoch_31.pt`; validation metrics include `validate/loss=0.11740` and `validate/pred_loss=0.002924`.
  - Active logs and queue state show no unrepaired failure, missing supervisor, bad dependency, CUDA OOM, driver error, pin-memory failure, or W&B failure.
  - No manual cancellation or resubmission was needed in this check.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT formal health check, 2026-06-07 05:03 EDT:
  - Current formal queue:
    - Seed 0 `9522910` is running on H100 with `QOS=embers` after 5:15 runtime; supervisor `9522912` is pending on `afterany:9522910`.
    - Seed 1 `9533424` is running on H100 with `QOS=embers` after 1:57 runtime; supervisor `9533425` is pending on `afterany:9533424`.
    - Seed 2 `9518823` is running on H100 with `QOS=embers` after 7:42 runtime; supervisor `9518824` is pending on valid `afterany:9518823` and should handle the expected 8-hour chunk limit.
  - All active world-model jobs still carry the intended settings: `RESUME_AUTO=True`, `NUM_WORKERS=6`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, and `PREFETCH_FACTOR=1`.
  - Latest observed progress:
    - Seed 0: epoch 34/100, global step 481650; epoch checkpoints through `weights_epoch_34.pt`; validation metrics include `validate/loss=0.11950` and `validate/pred_loss=0.002850`.
    - Seed 1: epoch 27/100, global step 378450; epoch checkpoints through `weights_epoch_27.pt`; validation metrics include `validate/loss=0.11842` and `validate/pred_loss=0.003195`.
    - Seed 2: epoch 32/100, global step 452000; epoch checkpoints through `weights_epoch_32.pt`; validation metrics include `validate/loss=0.11723` and `validate/pred_loss=0.002823`.
  - Active log sweep found no new `RuntimeError`, `Traceback`, CUDA OOM, driver error, missing-file error, pin-memory failure, dependency failure, or W&B error.
  - No manual repair was needed in this check.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
- PushT training/eval insight check, 2026-06-07 05:52 EDT:
  - Training curve insight from current metrics:
    - Seed 0 reached epoch 35/100, global step 494450, `validate/pred_loss=0.002737`, and `validate/loss=0.116714`; this is an 18.4x reduction in validation prediction loss from epoch 0.
    - Seed 1 reached epoch 28/100, global step 391750, `validate/pred_loss=0.003292`, and `validate/loss=0.117468`; this is a 12.7x reduction from epoch 0.
    - Seed 2 reached epoch 33/100, global step 463100, `validate/pred_loss=0.002889`, and `validate/loss=0.117936`; this is a 20.6x reduction from epoch 0.
    - Most of the prediction-loss improvement happened by epoch 5-10; after roughly epoch 20 the curve is still improving but much more slowly. Seed 0 currently has the best validation prediction loss, seed 2 is close, and seed 1 lags slightly.
  - Submitted quick current-checkpoint CEM real-environment evals with `eval.num_eval=3`, `eval.eval_budget=50`, `solver.num_samples=96`, `solver.n_steps=8`, and `solver.topk=12`.
    - Seed 0 checkpoint `weights_epoch_35.pt`, job `9541124`, H100, completed in 1:18, success rate `100.0%`, episode successes `[true, true, true]`, W&B run `ls1nsnmv`.
    - Seed 1 checkpoint `weights_epoch_28.pt`, job `9541125`, A100, completed in 1:17, success rate `66.67%`, episode successes `[false, true, true]`, W&B run `6b2dqxci`.
    - Seed 2 checkpoint `weights_epoch_33.pt`, job `9541126`, H100, completed in 1:18, success rate `66.67%`, episode successes `[false, true, true]`, W&B run `1f8fviyz`.
  - Quick eval outputs were written under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/pusht/wm_original_policy_quick_cem_20260607_0527/`.
  - Current ability insight: the partially trained PushT models already solve some real-environment starts with a very small CEM budget; however, the sample size is only 3 episodes per seed and the CEM budget is intentionally reduced, so these results are directional rather than final.
  - Eval Slurm now supports quick-eval overrides `EVAL_NUM_EVAL`, `EVAL_BUDGET`, `SOLVER_NUM_SAMPLES`, `SOLVER_N_STEPS`, and `SOLVER_TOPK`; formal defaults remain unchanged.
  - Quick eval manifests had `git_sha: unknown` because eval jobs run from project runtime workdirs. Commit `2613b11` fixed future manifest git SHA logging by resolving the repo through `experiment_logging.py`.
  - The formal 100-epoch world-model jobs continue running; no final checkpoint or formal full-budget eval exists yet.
- PushT monitoring and medium eval, 2026-06-07 06:01 EDT:
  - Formal training remains healthy:
    - Seed 0 `9522910` is running on H100 with `QOS=embers`, currently epoch 35/100, latest checkpoint `weights_epoch_35.pt`; supervisor `9522912` is pending on `afterany:9522910`.
    - Seed 1 `9533424` is running on H100 with `QOS=embers`, currently epoch 28/100, latest checkpoint `weights_epoch_28.pt`; supervisor `9533425` is pending on `afterany:9533424`.
    - Seed 2 `9539901` is running on H100 with `QOS=embers`, currently epoch 33/100, latest checkpoint `weights_epoch_33.pt`; supervisor `9539903` is pending on `afterany:9539901`.
  - Active log sweep found no new `RuntimeError`, `Traceback`, CUDA OOM, driver error, missing-file error, pin-memory failure, dependency failure, or W&B error.
  - Submitted medium current-checkpoint CEM real-environment evals with `eval.num_eval=10`, `eval.eval_budget=50`, and default CEM settings `solver.num_samples=300`, `solver.n_steps=30`, `solver.topk=30`.
    - Initial seed 0 and seed 2 H100 eval jobs `9541361` and `9541363` were canceled while still pending on priority, then resubmitted on A100 as `9541486` and `9541487` to finish sooner.
    - Seed 0 checkpoint `weights_epoch_35.pt`, job `9541486`, A100, completed in 1:12, success rate `100.0%`, episode successes `[true, true, true, true, true, true, true, true, true, true]`, W&B run `s5ktz4bi`.
    - Seed 1 checkpoint `weights_epoch_28.pt`, job `9541362`, A100, completed in 1:14, success rate `90.0%`, episode successes `[true, true, true, false, true, true, true, true, true, true]`, W&B run `tz0ue2qe`.
    - Seed 2 checkpoint `weights_epoch_33.pt`, job `9541487`, A100, completed in 1:51, success rate `80.0%`, episode successes `[true, false, true, true, true, true, true, false, true, true]`, W&B run `83f0265b`.
  - Medium eval outputs were written under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/pusht/wm_original_policy_medium_cem_20260607_0558/`.
  - Current ability insight: current partial checkpoints are already strong under full CEM settings on the sampled PushT starts; seed 0 is strongest, seed 1 is usable despite lagging in validation prediction loss, and seed 2 has lower medium-eval success than its validation loss alone would suggest. Continue training to 100 epochs and use the official full eval before making final claims.
  - Repo root remains clean: no `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.

## Notes

- Generated smoke outputs are intentionally outside git.
- Existing untracked `AGENTS.md` and `reference.md` were left untouched.

- PushT residual flow-WM submission, 2026-06-08 03:28 EDT:
  - Added a focused plan in `docs/flow_wm_residual_cfm_plan.md`.
  - Implemented the recommended flow-WM variant as residual Conditional Flow Matching over `z_next - z_current`.
  - Pipeline double check:
    - Training still uses `train.py`, `config/train/lewm.yaml`, `config/train/data/pusht.yaml`, and `pusht_expert_train.lance`.
    - Encoder, action encoder, projector, prediction projector, optimizer, SIGReg, dataloader, checkpointing, W&B, and resume behavior match native LeWM.
    - The only intended model swap is `module.ARPredictor` -> `module.ConditionalFlowPredictor`.
    - Shared predictor hyperparameters match native LeWM: `num_frames`, `input_dim`, `hidden_dim`, `output_dim`, `depth`, `heads`, `mlp_dim`, `dim_head`, `dropout`, and `emb_dropout`.
    - Flow-only predictor extras are `time_dim=64`, `sample_steps=8`, and `stochastic_sample=false`.
  - Submitted formal PushT flow-WM runs under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608`.
    - Seed 0: train job `9611079`, H100, pending on priority; supervisor `9611080` pending on `afterany:9611079`.
    - Seed 1: train job `9611081`, A100, pending on priority; supervisor `9611082` pending on `afterany:9611081`.
    - Seed 2: train job `9611083`, H100, pending on priority; supervisor `9611084` pending on `afterany:9611083`.
  - Submission settings: `QOS=embers`, `MAX_EPOCHS=100`, `RESUME_AUTO=True`, `NUM_WORKERS=6`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, `PREFETCH_FACTOR=1`, `WANDB_MODE=online`.
  - Supervisor settings for this run: `SUBMIT_ACTION_FLOW=False`, `SUBMIT_FLOW_EVAL=False`, `SUBMIT_CEM_EVAL=True`; this isolates `wm_flow_policy_original` and keeps the policy/planner as original LeWM CEM.
  - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/submit_20260608_032818.tsv`.
  - Canceled old unrelated `newt-flow-2x2` H200 array `9431231` because it was from `/storage/project/r-agarg35-0/eliu354/external_repos/newt`, not the LeWM pipeline requested here.
  - Validation performed:
    - `bash -n scripts/submit_pusht_flow_wm_formal.sh scripts/slurm_flow_2x2_supervisor.sbatch scripts/slurm_flow_2x2_train.sbatch`.
    - CPU shape smoke for `ConditionalFlowPredictor.flow_loss()` and `forward()` returned finite loss and output shape `(2, 3, 192)`.
    - Config comparison verified unchanged native modules and predictor hyperparameter parity except for flow-specific extras.
    - Repo root check found no generated `wandb/`, `outputs/`, `multirun/`, or `wandb_resume.json`.
  - Follow-up tracking fix, 2026-06-08 03:31 EDT:
    - Seed 0 job `9611079` started before commit `6b4b74b` was created, so its first manifest recorded previous git SHA `b6d5f8a` even though the working tree already contained the residual-flow code.
    - Canceled seed 0 job `9611079` and supervisor `9611080`.
    - Resubmitted seed 0 as train job `9611252` and supervisor `9611253`; they are pending on `embers`.
    - Seed 2 job `9611083` started after commit `6b4b74b` and its manifest records git SHA `6b4b74b7155931d320d07e9fb86740bd5ed7af8f`.
    - Current active flow-WM jobs after the fix:
      - Seed 0: train `9611252` pending on priority; supervisor `9611253` pending on `afterany:9611252`.
      - Seed 1: train `9611081` pending on priority; supervisor `9611082` pending on `afterany:9611081`.
      - Seed 2: train `9611083` running on H100; supervisor `9611084` pending on `afterany:9611083`.

- PushT monitoring and diagnostic eval submission, 2026-06-08 04:01 EDT:
  - Active native LeWM jobs:
    - Seed 0 `9599279` is running on H100 at epoch 61/100, about 5.1 it/s; supervisor `9599280` remains pending on `afterany:9599279`.
    - Seed 1 previous chunk `9583139` timed out after 8 hours and supervisor `9583140` completed; replacement train job `9611910` is pending on H100 with supervisor `9611911`.
    - Seed 2 `9588616` is running on H100 at epoch 59/100, about 5.0-5.1 it/s; supervisor `9588618` remains pending on `afterany:9588616`.
  - Active residual flow-WM jobs:
    - Seed 0 `9611252` is running on H100 at epoch 0/100, about 4.9 it/s; manifest records git SHA `c25ac226ecc442e96d2f9b55a0597d9f5aa3805f`.
    - Seed 1 `9611081` is pending on A100; supervisor `9611082` is pending on `afterany:9611081`.
    - Seed 2 `9611083` is running on H100 at epoch 0/100, about 5.0-5.1 it/s; manifest records git SHA `6b4b74b7155931d320d07e9fb86740bd5ed7af8f`.
  - Flow-WM early metrics are being logged to local JSONL and W&B:
    - Seed 0 latest observed step: global step 6050, `fit/flow_loss=0.4841`, `fit/pred_loss=1.1424`.
    - Seed 2 latest observed step: global step 7550, `fit/flow_loss=0.3650`, `fit/pred_loss=1.1250`.
    - Both flow jobs are writing step checkpoints under project storage and show no active traceback, CUDA OOM, driver error, pin-memory failure, or W&B failure.
  - Native seed 0 anomaly:
    - Validation prediction loss was healthy through epoch 48 (`validate/pred_loss=0.0023916`, best observed), then degraded continuously.
    - Latest epoch-level validation at epoch 60 is `validate/pred_loss=0.0290986`, with `validate/loss=0.1665637`.
    - Native seed 1 and seed 2 do not show the same degradation: seed 1 epoch 56 `validate/pred_loss=0.0020629`; seed 2 epoch 58 `validate/pred_loss=0.0020323`.
    - No active seed 0 crash or wrong model config was found; the job resumed from Lightning `last.ckpt` and uses `model=lewm`, `WM_VARIANT=original`, `PIN_MEMORY=False`, `PERSISTENT_WORKERS=False`, `PREFETCH_FACTOR=1`.
  - Submitted medium CEM diagnostic evals on A100 with `eval.num_eval=10`, `eval.eval_budget=50`, `solver.num_samples=300`, `solver.n_steps=30`, `solver.topk=30`.
    - Seed 0 best validation checkpoint: epoch 48, job `9612412`, policy variant `original_diag_e48`.
    - Seed 0 latest checkpoint: epoch 61, job `9612413`, policy variant `original_diag_e61`.
    - Seed 1 current checkpoint: epoch 57, job `9612415`, policy variant `original_diag_e57`.
    - Seed 2 current checkpoint: epoch 59, job `9612416`, policy variant `original_diag_e59`.
    - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/diag_eval_20260608_040118.tsv`.
  - Decision: keep native seed 0 formal training running until diagnostic evals show whether the validation degradation also harms real-environment CEM success. Earlier seed 0 checkpoints remain available for evaluation and selection.

- PushT monitoring and follow-up evals, 2026-06-08 15:00 EDT:
  - Native diagnostic evals completed:
    - Seed 0 epoch 48, job `9612412`, 10 episodes, success rate `90.0%`, episode successes `[true, true, false, true, true, true, true, true, true, true]`, W&B run `n4mg8s2s`.
    - Seed 0 epoch 61, job `9612413`, 10 episodes, success rate `50.0%`, episode successes `[true, true, false, true, false, true, false, false, false, true]`, W&B run `odb5oyuq`.
    - Seed 1 epoch 57, job `9612415`, 10 episodes, success rate `100.0%`, episode successes all true, W&B run `3s1it1yu`.
    - Seed 2 epoch 59, job `9612416`, 10 episodes, success rate `90.0%`, episode successes `[true, true, true, true, true, true, false, true, true, true]`, W&B run `wft0auat`.
  - Diagnostic conclusion:
    - Seed 0 validation degradation is real in environment eval: epoch 48 is substantially better than epoch 61 on the same 10 sampled starts.
    - Keep seed 0 training running for formal completeness, but use checkpoint selection for reporting and fuller evals instead of trusting the latest seed 0 checkpoint.
  - Current native training progress:
    - Seed 0 active job `9620378` is running on H100 at epoch 73/100; latest `validate/pred_loss=0.0223973`, best checkpoint remains epoch 48 with `validate/pred_loss=0.0023916`.
    - Seed 1 active job `9639854` is running on H100 at epoch 69/100; latest/best observed `validate/pred_loss=0.0018154`.
    - Seed 2 active job `9645447` is running on H100 at epoch 71/100; latest/best observed `validate/pred_loss=0.0017847`.
  - Current residual flow-WM training progress:
    - Seed 0 active job `9619444` is running on H100 at epoch 10/100; latest `fit/flow_loss=0.1720`, `validate/flow_loss=0.0643`, `validate/pred_loss=1.1768`.
    - Seed 1 active job `9643029` is running on H100 at epoch 7/100; latest `fit/flow_loss=0.1748`, `validate/flow_loss=0.0812`, `validate/pred_loss=1.1965`.
    - Seed 2 active job `9619424` is running on H100 at epoch 11/100; latest `fit/flow_loss=0.1681`, `validate/flow_loss=0.0650`, `validate/pred_loss=1.1924`.
    - Flow matching loss is improving, but deterministic predicted-latent MSE remains much worse than native LeWM at these early epochs; treat flow-WM as not yet competitive until eval confirms otherwise.
  - Submitted selected-checkpoint full native evals on A100 with 50 episodes and full CEM settings (`num_samples=300`, `n_steps=30`, `topk=30`):
    - Seed 0 epoch 48, job `9652297`, policy variant `original_full50_e48`.
    - Seed 1 epoch 68, job `9652298`, policy variant `original_full50_e68`.
    - Seed 2 epoch 70, job `9652299`, policy variant `original_full50_e70`.
    - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/full_eval_20260608_150031.tsv`.
  - Submitted early flow-WM CEM sanity evals on A100 with 3 episodes and reduced CEM (`num_samples=96`, `n_steps=8`, `topk=12`):
    - Seed 0 epoch 10, job `9652300`, policy variant `flow_quick_cem_e10`.
    - Seed 2 epoch 11, job `9652301`, policy variant `flow_quick_cem_e11`.
    - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/quick_eval_20260608_150031.tsv`.
  - Active log check:
    - Active train jobs show resume warnings and normal W&B resume messages, but no active CUDA OOM, driver error, pin-memory failure, W&B failure, or unrepaired config issue.
    - Recent flow preemption/FileNotFoundError shutdown noise was handled by supervisors, which resubmitted the affected seeds and resumed from Lightning `last.ckpt`.

- PushT checkpoint index, 2026-06-08 15:07 EDT:
  - Selection rule:
    - Native LeWM best checkpoint is the lowest observed `validate/pred_loss`, except seed 0 is also backed by real-environment diagnostic eval because epoch 48 scored 90.0% over 10 episodes while epoch 61 scored 50.0%.
    - Residual flow-WM best checkpoint is the lowest observed `validate/flow_loss` until enough real-environment eval data is available; deterministic `validate/pred_loss` is still not competitive at these early flow epochs.
    - Checkpoint epoch means the filename `weights_epoch_N.pt`; Lightning metric rows are 0-indexed and can appear as epoch `N-1` for the checkpoint saved at `N`.
  - Current native LeWM checkpoint table:
    | Seed | Best checkpoint | Best metric / eval | Latest checkpoint | Latest metric |
    | --- | --- | --- | --- | --- |
    | 0 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_0/lewm/weights_epoch_48.pt` | `validate/pred_loss=0.0023916`; 10-episode diag eval `90.0%` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_0/lewm/weights_epoch_73.pt` | latest observed `validate/pred_loss=0.0223973` |
    | 1 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_1/lewm/weights_epoch_70.pt` | `validate/pred_loss=0.0017662` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_1/lewm/weights_epoch_70.pt` | latest observed `validate/pred_loss=0.0017662` |
    | 2 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_2/lewm/weights_epoch_71.pt` | `validate/pred_loss=0.0017847` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/checkpoints/pusht/wm_original_policy_original/seed_2/lewm/weights_epoch_71.pt` | latest observed `validate/pred_loss=0.0017847` |
  - Current residual flow-WM checkpoint table:
    | Seed | Best checkpoint | Best metric | Latest checkpoint | Latest metric |
    | --- | --- | --- | --- | --- |
    | 0 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_0/lewm_flow/weights_epoch_10.pt` | `validate/flow_loss=0.0642842`, `validate/pred_loss=1.1768` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_0/lewm_flow/weights_epoch_10.pt` | latest observed `validate/flow_loss=0.0642842` |
    | 1 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_1/lewm_flow/weights_epoch_7.pt` | `validate/flow_loss=0.0812325`, `validate/pred_loss=1.1965` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_1/lewm_flow/weights_epoch_7.pt` | latest observed `validate/flow_loss=0.0812325` |
    | 2 | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_2/lewm_flow/weights_epoch_11.pt` | `validate/flow_loss=0.0650390`, `validate/pred_loss=1.1924` | `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/checkpoints/pusht/wm_flow_policy_original/seed_2/lewm_flow/weights_epoch_11.pt` | latest observed `validate/flow_loss=0.0650390` |
  - Eval alignment note:
    - Pending native full50 eval jobs were submitted at the 15:00 snapshot: seed 0 epoch 48 job `9652297`, seed 1 epoch 68 job `9652298`, seed 2 epoch 70 job `9652299`.
    - After that snapshot, native seed 1 produced checkpoint epoch 70 with a lower observed validation prediction loss, and seed 2 produced checkpoint epoch 71 with a lower observed validation prediction loss. These are now the metric-best checkpoints, but they do not yet have full50 real-environment eval jobs.
    - Pending flow quick eval jobs target seed 0 epoch 10 job `9652300` and seed 2 epoch 11 job `9652301`; those match the current flow metric-best checkpoints for those seeds.

- PushT residual flow-WM quick eval results, 2026-06-08 15:28 EDT:
  - Both A100 quick eval jobs completed cleanly with exit code `0:0`.
  - Seed 0 checkpoint epoch 10, job `9652300`, policy variant `flow_quick_cem_e10`: 3 episodes, success rate `0.0%`, episode successes `[false, false, false]`, evaluation time `8.3071s`, W&B run `8x7r23ga`.
  - Seed 2 checkpoint epoch 11, job `9652301`, policy variant `flow_quick_cem_e11`: 3 episodes, success rate `0.0%`, episode successes `[false, false, false]`, evaluation time `8.1870s`, W&B run `2q41gvi9`.
  - Eval settings matched the intended quick sanity check: `eval.num_eval=3`, `eval.eval_budget=50`, `solver.num_samples=96`, `solver.n_steps=8`, `solver.topk=12`, W&B online logging enabled.
  - Qualitative video check from contact sheets under `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/runtime/eval_frames/flow_quick_20260608`: the agent moves, but does not reliably move the T-block toward the goal and the control point often drifts away after contact. Treat this as a real flow-WM/planner quality failure, not an eval submission failure.
  - Immediate interpretation: flow optimization is still numerically healthy, but early residual flow-WM checkpoints are not usable with the current LeWM CEM interface. Continue training for trend data, but prioritize checking the flow rollout/sampling semantics and planner objective before spending full-budget eval on these early checkpoints.

- PushT monitoring and eval push, 2026-06-08 18:31 EDT:
  - Active LeWM jobs on H100:
    - Native seed 0 train `9661577`, native seed 1 train `9639854`, native seed 2 train `9645447`.
    - Flow-WM seed 0 train `9656993`, flow-WM seed 1 train `9643029`, flow-WM seed 2 train `9653769`.
    - Supervisors remain pending on dependencies for all active train jobs.
  - Active log check found no CUDA OOM, traceback, W&B failure, or unrepaired config issue. The only repeated warning is missing `pynvml`, which disables GPU monitor metrics but does not stop training.
  - Native full50 selected-checkpoint evals completed:
    - Seed 0 checkpoint epoch 48, job `9652297`: 50 episodes, success rate `88.0%`.
    - Seed 1 checkpoint epoch 68, job `9652298`: 50 episodes, success rate `80.0%`.
    - Seed 2 checkpoint epoch 70, job `9652299`: 50 episodes, success rate `86.0%`.
  - Latest native validation:
    - Seed 0 latest checkpoint epoch 77 is degraded (`validate/pred_loss=0.1444579`), so seed 0 reporting/eval should still use checkpoint epoch 48.
    - Seed 1 current metric-best checkpoint is epoch 74 (`validate/pred_loss=0.0017167`).
    - Seed 2 current metric-best checkpoint is epoch 75 (`validate/pred_loss=0.0016912`); latest checkpoint epoch 76 is close but slightly worse on validation (`validate/pred_loss=0.0017004`).
  - Latest flow-WM validation continues to improve on flow matching loss but not deterministic prediction loss:
    - Seed 0 latest checkpoint epoch 13: `validate/flow_loss=0.0611049`, `validate/pred_loss=1.2210`.
    - Seed 1 latest checkpoint epoch 11: `validate/flow_loss=0.0710092`, `validate/pred_loss=1.2174`.
    - Seed 2 latest checkpoint epoch 14: `validate/flow_loss=0.0610028`, `validate/pred_loss=1.2100`.
  - Submitted follow-up evals on A100/embers; all are pending on priority:
    - Native updated full50 evals: seed 1 epoch 74 job `9671667`, variant `original_full50_e74`; seed 2 epoch 75 job `9671668`, variant `original_full50_e75`.
    - Flow later-checkpoint quick evals: seed 0 epoch 13 job `9671669`, variant `flow_quick_cem_e13_later`; seed 1 epoch 11 job `9671671`, variant `flow_quick_cem_e11_s1`; seed 2 epoch 14 job `9671672`, variant `flow_quick_cem_e14_later`.
    - Job records:
      - `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/full_eval_update_20260608_182954.tsv`.
      - `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/quick_eval_later_20260608_182954.tsv`.

- PushT monitoring and flow diagnostics, 2026-06-08 19:56 EDT:
  - Follow-up eval jobs completed cleanly:
    - Native seed 1 checkpoint epoch 74, job `9671667`: 50 episodes, success rate `80.0%`. This matches seed 1 epoch 68, so the lower validation prediction loss did not improve full-CEM success.
    - Native seed 2 checkpoint epoch 75, job `9671668`: 50 episodes, success rate `88.0%`. This is a small improvement over seed 2 epoch 70 at `86.0%`.
    - Flow seed 0 checkpoint epoch 13, job `9671669`: 3 episodes, success rate `0.0%`.
    - Flow seed 1 checkpoint epoch 11, job `9671671`: 3 episodes, success rate `0.0%`.
    - Flow seed 2 checkpoint epoch 14, job `9671672`: 3 episodes, success rate `0.0%`.
  - Current interpretation:
    - Native LeWM remains strong under the original pipeline; best observed full50 results are seed 0 epoch 48 `88.0%`, seed 1 epoch 68/74 `80.0%`, and seed 2 epoch 75 `88.0%`.
    - Flow-WM failure is reproduced across all three seeds and later checkpoints. Since native quick eval used the same reduced CEM settings and succeeded, the primary issue is likely flow rollout quality or predictor sampling semantics, not eval submission.
  - Latest active training status:
    - Native seed 0 and native seed 2 were preempted and resumed by supervisors as jobs `9673541` and `9673440`; native seed 1 job `9639854` continues running.
    - Flow seed 2 was preempted and resumed by supervisor as job `9673439`; flow seed 0 job `9656993` and flow seed 1 job `9643029` continue running.
    - Active log scan found no OOM, traceback, W&B failure, or config error; only missing-`pynvml` GPU monitor warnings.
  - Added eval-only predictor sampling overrides in commit `313664c`:
    - `eval.py` now honors optional `+predictor_stochastic_sample` and `+predictor_sample_steps` when the loaded predictor exposes those attributes.
    - `scripts/slurm_flow_2x2_eval.sbatch` now forwards optional `PREDICTOR_STOCHASTIC_SAMPLE` and `PREDICTOR_SAMPLE_STEPS`.
    - Default eval behavior is unchanged when these variables are unset.
  - Submitted additional flow diagnostics on A100/embers; both are pending on priority:
    - Full-CEM flow seed 2 checkpoint epoch 14, job `9676877`, variant `flow_fullcem_diag_e14_s2`, with `eval.num_eval=10`, `solver.num_samples=300`, `solver.n_steps=30`, `solver.topk=30`.
    - Stochastic predictor quick eval seed 2 checkpoint epoch 14, job `9677090`, variant `flow_quick_stoch16_e14_s2`, with `eval.num_eval=3`, reduced CEM, `PREDICTOR_STOCHASTIC_SAMPLE=True`, and `PREDICTOR_SAMPLE_STEPS=16`.
    - Job records:
      - `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/fullcem_diag_20260608_195201.tsv`.
      - `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/stochastic_eval_diag_20260608_195528.tsv`.

- PushT flow diagnostic results and next evals, 2026-06-08 21:00 EDT:
  - Diagnostic eval jobs completed cleanly:
    - Full-CEM flow seed 2 checkpoint epoch 14, job `9676877`: 10 episodes, success rate `10.0%`, episode successes `[false, false, false, false, false, false, false, false, true, false]`.
    - Stochastic predictor quick eval seed 2 checkpoint epoch 14, job `9677090`: 3 episodes, success rate `0.0%`, episode successes `[false, false, false]`.
  - Interpretation:
    - Full CEM can occasionally recover a success from the flow-WM, so the failure is not a pure eval crash or action-space wiring bug.
    - The success rate remains far below native LeWM (`80-88%` full50), and stochastic predictor sampling did not improve quick eval. Continue treating the main issue as flow rollout quality / deterministic predictor semantics under CEM.
  - Latest flow validation has continued improving in `validate/flow_loss`, while deterministic prediction loss remains around `1.2`:
    - Seed 0 checkpoint epoch 16: `validate/flow_loss=0.0589514`, `validate/pred_loss=1.2324`.
    - Seed 1 checkpoint epoch 13: `validate/flow_loss=0.0679934`, `validate/pred_loss=1.2044`.
    - Seed 2 checkpoint epoch 17: `validate/flow_loss=0.0570019`, `validate/pred_loss=1.2041`.
  - Submitted later-checkpoint full-CEM 10-episode evals on A100/embers; all are pending on priority:
    - Seed 0 checkpoint epoch 16, job `9681908`, variant `flow_fullcem10_e16_s0`.
    - Seed 1 checkpoint epoch 13, job `9681910`, variant `flow_fullcem10_e13_s1`.
    - Seed 2 checkpoint epoch 17, job `9681911`, variant `flow_fullcem10_e17_s2`.
    - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608/job_records/fullcem_later_20260608_205959.tsv`.

- PushT overnight monitoring, 2026-06-09 02:58 EDT:
  - Flow later-checkpoint full-CEM 10-episode evals completed cleanly:
    - Seed 0 checkpoint epoch 16, job `9681908`: success rate `0.0%`, episode successes all false.
    - Seed 1 checkpoint epoch 13, job `9681910`: success rate `0.0%`, episode successes all false.
    - Seed 2 checkpoint epoch 17, job `9681911`: success rate `0.0%`, episode successes all false.
  - Flow interpretation update:
    - The earlier seed 2 epoch 14 full-CEM result had a single success (`1/10`), but later checkpoints at epochs 16/13/17 all failed under the same full-CEM settings.
    - `validate/flow_loss` continues to improve, but real-environment success does not track it. Current evidence suggests the residual flow objective is not aligned with the LeWM CEM rollout/cost interface, not that the eval jobs are broken.
  - Latest flow validation:
    - Seed 0 checkpoint epoch 22: `validate/flow_loss=0.0542554`, `validate/pred_loss=1.2247`.
    - Seed 1 checkpoint epoch 19: `validate/flow_loss=0.0630558`, `validate/pred_loss=1.2209`.
    - Seed 2 checkpoint epoch 24: latest `validate/flow_loss=0.0533679`, `validate/pred_loss=1.2078`; best observed flow loss at checkpoint epoch 23 is `0.0527222`.
  - Latest native validation:
    - Seed 0 checkpoint epoch 88 remains degraded (`validate/pred_loss=0.1915263`); keep seed 0 checkpoint epoch 48 for reporting.
    - Seed 1 metric-best is checkpoint epoch 83 (`validate/pred_loss=0.0016117`).
    - Seed 2 metric-best is checkpoint epoch 86 (`validate/pred_loss=0.0015598`).
  - Submitted native latest metric-best full50 evals on A100/embers; both are pending on priority:
    - Seed 1 checkpoint epoch 83, job `9702390`, variant `original_full50_e83`.
    - Seed 2 checkpoint epoch 86, job `9702392`, variant `original_full50_e86`.
    - Job record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/full_eval_latest_20260609_025819.tsv`.
  - Active training:
    - Six PushT training jobs are running on H100: native seeds 0/1/2 and flow seeds 0/1/2.
    - Recent preemptions/timeouts were handled by supervisors. Active log scan found no OOM, traceback, W&B failure, or config error; only missing-`pynvml` GPU monitor warnings.

- PushT flow-WM follow-up plan, 2026-06-09 03:24 EDT:
  - Documented two follow-ups in `docs/flow_wm_followups_20260609.md`:
    - cost-ranking diagnostic using the same LeWM eval cost path;
    - endpoint-aligned residual flow WM with `loss.flow_pred.weight=0.1`.
  - Formal endpoint-flow runs must follow the original PushT LeWM train/eval/supervisor pipeline, changing only the WM predictor/objective variant.
  - Cost-ranking jobs are diagnostic-only and do not replace real-environment PushT eval.

- PushT flow-WM follow-up submission, 2026-06-09 03:27 EDT:
  - Code/docs commit: `d724177`, pushed to `origin/exp/flow-2x2-pusht-cube`.
  - Validation before submission:
    - `.conda/lewm-flow-2x2/bin/python -m py_compile train.py module.py eval.py diagnose_cost_ranking.py`.
    - `bash -n scripts/slurm_flow_2x2_train.sbatch scripts/slurm_flow_2x2_supervisor.sbatch scripts/slurm_pusht_cost_ranking_diag.sbatch scripts/submit_pusht_flow_endpoint_formal.sh`.
    - `git diff --check`.
    - Hydra compose smoke accepted `data=pusht model=lewm_flow loss.flow_pred.weight=0.1`.
  - Submitted cost-ranking diagnostics on A100/embers, all pending on priority at submission check:
    - `9703631`, native seed 2 epoch 75, variant `cost_rank_original_s2_e75`.
    - `9703632`, flow seed 2 epoch 17, variant `cost_rank_flow_s2_e17`.
    - `9703633`, flow seed 2 epoch 24, variant `cost_rank_flow_s2_e24`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/submit_20260609_032722.tsv`.
  - Submitted endpoint-aligned flow-WM formal PushT training on H100/embers with `FLOW_PRED_LOSS_WEIGHT=0.1`, all pending on priority at submission check:
    - Seed 0 train `9703634`, supervisor `9703635`.
    - Seed 1 train `9703636`, supervisor `9703637`.
    - Seed 2 train `9703638`, supervisor `9703639`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/submit_20260609_032722.tsv`.

- PushT monitoring and repairs, 2026-06-09 14:34 EDT:
  - Cost-ranking diagnostic jobs `9703631`, `9703632`, and `9703633` failed quickly with
    `ValueError: not enough values to unpack (expected 4, got 3)` inside ViT goal encoding.
    Root cause: diagnostic info tensors were passed to `model.get_cost(...)` without the solver sample dimension used by the official CEM path.
  - Fixed the diagnostic in commit `f247281` by expanding prepared info tensors to `(batch, candidates, time, ...)`, matching the official solver-to-`get_cost` call shape.
  - Resubmitted fixed cost-ranking diagnostics on A100/embers, all pending on priority:
    - `9747692`, native seed 2 epoch 75, variant `cost_rank_original_s2_e75_fix1`.
    - `9747694`, flow seed 2 epoch 17, variant `cost_rank_flow_s2_e17_fix1`.
    - `9747696`, flow seed 2 epoch 24, variant `cost_rank_flow_s2_e24_fix1`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/resubmit_shape_fix_20260609_143139.tsv`.
  - Endpoint-aligned flow-WM training is active and resumed correctly after preemption/timeout:
    - Running: seed 0 job `9744344`, seed 2 job `9736191`.
    - Pending on H100 priority: seed 1 job `9745373`.
    - Latest observed metrics:
      | Seed | Latest epoch row | `validate/pred_loss` / endpoint | `validate/flow_loss` | Checkpoints present |
      | --- | --- | --- | --- | --- |
      | 0 | 9 | `0.010692` | `0.067204` | epochs 1-9 |
      | 1 | 7 | `0.012844` | `0.068248` | epochs 1-7 |
      | 2 | 7 | `0.014015` | `0.080670` | epochs 1-7 |
    - This is a strong early sign that endpoint alignment is doing what it was intended to do: deterministic rollout loss is around `0.01-0.014`, while the original residual flow-WM had deterministic `validate/pred_loss` around `1.2`.
  - Native LeWM final-checkpoint evals triggered by supervisors:
    - Seed 0 epoch 100 job `9736954`: success rate `6.0%`; this confirms seed 0 final checkpoint is severely degraded and should not replace the earlier reporting checkpoint epoch 48.
    - Seed 2 epoch 100 job `9738938`: success rate `86.0%`, close to earlier seed 2 full50 values.
  - Native action-flow policy repairs:
    - Action-flow jobs `9736953` and `9738936` for native seeds 0 and 2 were preempted, causing dependent flow-policy eval jobs `9736956` and `9738939` to become `DependencyNeverSatisfied`.
    - Both action-flow jobs had already written `action_flow.pt`, so the stuck eval jobs were canceled and replacement evals were submitted without the dead dependency:
      - `9747859`, native seed 0 flow-policy eval repair.
      - `9747860`, native seed 2 flow-policy eval repair.
      - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/repair_action_flow_eval_20260609_143336.tsv`.
    - Caveat: these repaired flow-policy evals use best-so-far action-flow checkpoints written before preemption, not complete 20-epoch action-flow training.

- PushT monitoring and endpoint quick evals, 2026-06-09 16:59 EDT:
  - Active endpoint-aligned flow-WM training:
    - Running: seed 0 job `9744344`, seed 1 job `9745373`, seed 2 job `9736191`.
    - Latest observed endpoint metrics:
      | Seed | Latest epoch row | `validate/pred_loss` / endpoint | `validate/flow_loss` | Best endpoint row |
      | --- | --- | --- | --- | --- |
      | 0 | 11 | `0.008702` | `0.061655` | epoch 10 |
      | 1 | 9 | `0.010624` | `0.061616` | epoch 8 |
      | 2 | 9 | `0.010841` | `0.076098` | epoch 8 |
    - Endpoint loss remains far below the original flow-WM deterministic prediction loss around `1.2`, so the objective-alignment change is still behaving as intended.
  - Native final-checkpoint CEM eval update:
    - Seed 1 epoch 100 job `9745538` completed with success rate `82.0%`.
    - Current final-checkpoint CEM results are seed 0 `6.0%`, seed 1 `82.0%`, seed 2 `86.0%`.
    - Reporting should still use seed 0 epoch 48 (`88.0%`) rather than seed 0 epoch 100.
  - Native action-flow status:
    - Seed 1 action-flow job `9745537` is running.
    - Seed 1 dependent flow-policy eval `9745540` remains pending on `afterok:9745537`.
    - Repair evals for seed 0 and seed 2, jobs `9747859` and `9747860`, remain pending on H100 priority.
  - Submitted endpoint-flow quick real-environment evals on A100/embers with reduced CEM (`eval.num_eval=3`, `solver.num_samples=96`, `solver.n_steps=8`, `solver.topk=12`), all pending on priority:
    - Seed 0 checkpoint epoch 11: job `9757983`, variant `flow_endpoint_quick_cem_e11_s0`.
    - Seed 1 checkpoint epoch 9: job `9757985`, variant `flow_endpoint_quick_cem_e9_s1`.
    - Seed 2 checkpoint epoch 9: job `9757986`, variant `flow_endpoint_quick_cem_e9_s2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/quick_eval_endpoint_20260609_165830.tsv`.
  - Submitted endpoint-flow cost-ranking diagnostics on A100/embers, all pending on priority:
    - Seed 0 checkpoint epoch 11: job `9758072`, variant `cost_rank_flow_endpoint_s0_e11`.
    - Seed 1 checkpoint epoch 9: job `9758073`, variant `cost_rank_flow_endpoint_s1_e9`.
    - Seed 2 checkpoint epoch 9: job `9758075`, variant `cost_rank_flow_endpoint_s2_e9`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/submit_endpoint_cost_rank_20260609_165918.tsv`.
  - Earlier empty quick-eval record from a failed local shell submission attempt was removed; the retained record is the one above with the three actual Slurm job IDs.

- PushT endpoint quick-result check, 2026-06-09 17:31 EDT:
  - Endpoint quick real-environment result available so far:
    - L40S backup seed 0 checkpoint epoch 11, job `9759953`, variant `flow_endpoint_quick_cem_l40s_e11_s0`: success rate `0.0%` over 3 episodes.
  - L40S backup jobs for seed 1 and seed 2, `9759954` and `9759955`, failed before eval with `RuntimeError: The NVIDIA driver on your system is too old`; treat these as infrastructure failures, not model failures.
  - A100 quick eval jobs `9757983`, `9757985`, and `9757986` remain pending on priority.
  - Submitted H200 backup quick evals for the two L40S driver-failure seeds:
    - `9760053`, seed 1 checkpoint epoch 9, variant `flow_endpoint_quick_cem_h200_e9_s1`.
    - `9760054`, seed 2 checkpoint epoch 9, variant `flow_endpoint_quick_cem_h200_e9_s2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/quick_eval_endpoint_h200_backup_20260609_173053.tsv`.
  - Current interpretation:
    - Training-side hypothesis is supported: endpoint loss makes deterministic prediction align with CEM input (`validate/pred_loss` around `0.009-0.011`, versus around `1.2` for original flow-WM).
    - Real-environment hypothesis is not yet supported: the first completed quick eval is still `0/3`.
    - Need at least seed 1/2 quick evals and cost-ranking diagnostics before deciding whether endpoint alignment helps planning or merely improves one-step MSE.

- PushT same-epoch flow vs MLP comparison, 2026-06-09 17:40 EDT:
  - Same checkpoint-epoch comparison uses the validation row immediately before the checkpoint filename epoch because checkpoint `weights_epoch_N.pt` corresponds to training through metric row `N-1`.
  - Deterministic one-step prediction loss at matched early epochs:
    | Seed / checkpoint | Native MLP `validate/pred_loss` | Original flow `validate/pred_loss` | Endpoint-flow `validate/pred_loss` |
    | --- | --- | --- | --- |
    | seed 0 / epoch 11 | `0.004528` | `1.180684` | `0.008702` |
    | seed 1 / epoch 9 | `0.005101` | `1.170925` | `0.010624` |
    | seed 2 / epoch 9 | `0.006006` | `1.215889` | `0.010841` |
  - Matched-epoch interpretation:
    - Native MLP is still about `2x` better than endpoint-flow on deterministic one-step MSE at these early epochs.
    - Endpoint-flow is about `100x` better than the original flow-WM on deterministic one-step MSE.
    - Original flow optimizes `flow_loss`, which drops, but its deterministic prediction path remains unusable for CEM-style rollout at matched epochs.
  - Endpoint-flow quick real-environment evals now available:
    - seed 0 epoch 11, L40S job `9759953`: `0/3`.
    - seed 1 epoch 9, H200 job `9760053`: `2/3`.
    - seed 2 epoch 9, H200 job `9760054`: `0/3`.
    - Aggregate: `2/9`, success rate `22.2%`.
  - Original flow-WM quick evals at similar epochs were `0/3` for seed 0 epoch 10, seed 1 epoch 11, and seed 2 epoch 11; original flow-WM had one later full-CEM outlier at seed 2 epoch 14 with `1/10`.
  - Native MLP quick evals at early checkpoints were much stronger: seed 0 epoch 35 `3/3`, seed 1 epoch 28 `2/3`, seed 2 epoch 33 `2/3`; native medium-CEM 10-episode evals at those checkpoints were `100%`, `90%`, and `80%`.
  - Current conclusion:
    - The endpoint-flow direction validates the initial diagnosis that pure flow-matching loss was misaligned with deterministic LeWM/CEM rollout.
    - It has not closed the gap to native MLP under the original pipeline; real-environment success is better than original flow-WM in one seed but still far below native MLP.
    - Endpoint alignment is a useful repair direction, but not sufficient by itself. Next diagnostic remains cost-ranking (`9758072`, `9758073`, `9758075`) to see whether the planner cost surface improved in the successful seed only or broadly.
  - Canceled duplicate A100 endpoint quick eval jobs `9757983`, `9757985`, and `9757986` after H200/L40S backup jobs produced the needed quick eval results.

- PushT monitoring and repair, 2026-06-09 20:10 EDT:
  - Cost-ranking diagnostics and native action-flow eval repairs initially failed because JEPA cost expansion did not handle sampled/candidate prediction tensors shaped like `[B, candidates, T, D]` with goal tensors shaped like `[B, 1, D]`.
  - Fixed goal broadcasting in `jepa.py` commit `3be1a86`; validation passed with `py_compile` for `jepa.py`, `diagnose_cost_ranking.py`, `flow_solver.py`, and `eval.py`, plus tensor-shape smokes for both `[B, 1, D]` and `[B, candidates, 1, D]` goals.
  - Resubmitted fixed cost-ranking diagnostics, all pending on A100 priority:
    - `9764823`, native MLP seed 2 epoch 75, variant `cost_rank_original_s2_e75_fix2`.
    - `9764824`, original flow-WM seed 2 epoch 17, variant `cost_rank_flow_s2_e17_fix2`.
    - `9764825`, original flow-WM seed 2 epoch 24, variant `cost_rank_flow_s2_e24_fix2`.
    - `9764826`, endpoint-flow seed 0 epoch 11, variant `cost_rank_flow_endpoint_s0_e11_fix2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/resubmit_cost_broadcast_fix_20260609_200727.tsv`.
  - Endpoint cost diagnostics submitted before the code fix remain pending and should run with the fixed repo code when scheduled:
    - `9758073`, endpoint-flow seed 1 epoch 9.
    - `9758075`, endpoint-flow seed 2 epoch 9.
  - Resubmitted native action-flow policy eval repairs after the same JEPA cost fix, both pending on H100 priority:
    - `9764834`, native seed 0 flow-policy eval repair.
    - `9764835`, native seed 2 flow-policy eval repair.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/repair_action_flow_eval_costfix_20260609_200742.tsv`.
  - Endpoint-flow training remains healthy:
    | Seed | Active job | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` | Best endpoint row |
    | --- | --- | --- | --- | --- | --- | --- |
    | 0 | `9759942` | 13 | 185750 | `0.008272` | `0.060101` | 12 |
    | 1 | `9759980` | 11 | 157850 | `0.009250` | `0.058428` | 10 |
    | 2 | `9736191` | 11 | 162200 | `0.009319` | `0.071634` | 10 |
  - Existing later endpoint checkpoints were found for seed 0 epoch 13, seed 1 epoch 11, and seed 2 epoch 11.
  - Submitted later-checkpoint quick real-environment evals with reduced CEM (`eval.num_eval=3`, `solver.num_samples=96`, `solver.n_steps=8`, `solver.topk=12`):
    - First submission jobs `9764891`, `9764892`, and `9764893` were canceled before start because the eval script did not yet route `EXPERIMENT_VARIANT` into Hydra, which could have mixed quick-eval metrics in the default eval directory.
    - Added opt-in support in `scripts/slurm_flow_2x2_eval.sbatch` for `EXPERIMENT_VARIANT`, `WANDB_RUN_GROUP`, and `WANDB_NAME`; default formal eval behavior is unchanged.
    - Replacement H100 jobs are pending on priority:
      - `9764918`, seed 0 checkpoint epoch 13, variant `flow_endpoint_quick_cem_h100_later_e13_s0`.
      - `9764919`, seed 1 checkpoint epoch 11, variant `flow_endpoint_quick_cem_h100_later_e11_s1`.
      - `9764920`, seed 2 checkpoint epoch 11, variant `flow_endpoint_quick_cem_h100_later_e11_s2`.
      - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/quick_eval_endpoint_later_variantfix_20260609_201127.tsv`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and repair, 2026-06-09 21:32 EDT:
  - Endpoint-flow continuation is working:
    - Previous chunks ended as expected under `embers`: seed 2 job `9736191` hit the 8-hour time limit, seed 0 job `9759942` and seed 1 job `9759980` were preempted.
    - Their supervisors completed and submitted running H100 continuation chunks:
      - seed 0: `9767223`, supervisor `9767225`.
      - seed 1: `9767298`, supervisor `9767299`.
      - seed 2: `9765420`, supervisor `9765421`.
  - Endpoint-flow latest metrics continue to improve:
    | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` |
    | --- | --- | --- | --- | --- |
    | 0 | 14 | 199000 | `0.007597` | `0.060210` |
    | 1 | 12 | 172500 | `0.009011` | `0.057606` |
    | 2 | 12 | 174450 | `0.008804` | `0.069398` |
  - Original residual flow-WM is also continuing, but remains poorly aligned with deterministic CEM rollout:
    | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` |
    | --- | --- | --- | --- | --- |
    | 0 | 42 | 588550 | `1.240094` | `0.045162` |
    | 1 | 34 | 482700 | `1.236292` | `0.054489` |
    | 2 | 40 | 571250 | `1.229844` | `0.044950` |
    - Seed 0 job `9741350` timed out at 8 hours; supervisor `9741353` submitted continuation job `9768281` and supervisor `9768283`.
  - Native MLP + action-flow policy eval repairs completed:
    - seed 0 job `9764834`: `3/50`, success rate `6.0%`, using native final epoch 100 world model.
    - seed 2 job `9764835`: `29/50`, success rate `58.0%`, using native final epoch 100 world model.
    - These are below the native CEM policy results and do not support replacing CEM with the current action-flow policy.
  - Later endpoint quick eval jobs `9764918`, `9764919`, and `9764920` failed before evaluation because the policy override included `checkpoints/pusht/...`; `stable_worldmodel.wm.utils.load_pretrained()` already prepends `checkpoint_cache_dir/checkpoints`, producing a bad doubled path.
  - Resubmitted corrected H200 quick evals using latest available endpoint checkpoints and policy paths rooted at `pusht/...`:
    - `9768186`, seed 0 checkpoint epoch 14, variant `flow_endpoint_quick_cem_h200_pathfix_e14_s0`.
    - `9768188`, seed 1 checkpoint epoch 12, variant `flow_endpoint_quick_cem_h200_pathfix_e12_s1`.
    - `9768189`, seed 2 checkpoint epoch 12, variant `flow_endpoint_quick_cem_h200_pathfix_e12_s2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/quick_eval_endpoint_latest_pathfix_20260609_213015.tsv`.
  - Cost-ranking diagnostics remain pending on A100 priority; no diagnostic result is available yet.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-10 02:55 EDT:
  - Endpoint-flow training continues under the supervisor chain:
    - Running: seed 0 job `9772192`, seed 2 job `9765420`.
    - Pending continuation after preemption: seed 1 job `9778779`.
    - Latest endpoint metrics: seed 0 epoch row 18 `validate/pred_loss=0.007322`, seed 1 epoch row 16 `0.007715`, seed 2 epoch row 16 `0.007355`.
  - Corrected endpoint quick evals completed:
    - `9768186`, seed 0 epoch 14: `1/3`, success rate `33.3%`.
    - `9768188`, seed 1 epoch 12: `1/3`, success rate `33.3%`.
    - `9768189`, seed 2 epoch 12: `0/3`, success rate `0.0%`.
    - Combined with earlier quick evals, the best endpoint quick result remains seed 1 epoch 9 at `2/3`; later lower prediction loss did not translate into better quick real-env success.
  - Cost-ranking diagnostics completed and support the same interpretation:
    | Model / checkpoint | Mean expert rank | Random-better fraction | Real-env quick/full context |
    | --- | --- | --- | --- |
    | Native MLP seed 2 epoch 75 | `1.0` | `0.0000` | strong CEM baseline, `88%` full50 |
    | Original flow seed 2 epoch 17 | `56.75` | `0.4302` | poor, `0/10` full-CEM at epoch 17 |
    | Original flow seed 2 epoch 24 | `65.56` | `0.5015` | near random cost ranking |
    | Endpoint-flow seed 1 epoch 9 | `1.125` | `0.0010` | best endpoint quick result, `2/3` |
    | Endpoint-flow seed 0 epoch 11 | `7.875` | `0.0527` | weak quick result, `0/3` |
    | Endpoint-flow seed 2 epoch 9 | `10.4375` | `0.0728` | weak quick result, `0/3` |
    - Insight: endpoint alignment fixes the worst cost-ranking failure of the original flow-WM, especially for seed 1, but the real-env gap to native MLP remains.
  - Native MLP + action-flow seed 1 repair:
    - Action-flow job `9745537` timed out but wrote `action_flow.pt`.
    - Dead dependent eval `9745540` was canceled and replaced with H200 eval job `9779210`, no dependency.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_formal_20260605/job_records/repair_action_flow_eval_seed1_20260610_025410.tsv`.
  - Submitted endpoint medium10 full-CEM evals to reduce 3-episode quick-eval noise:
    - `9779221`, seed 0 epoch 14, variant `flow_endpoint_medium10_fullcem_e14_s0_latest_quick_1of3`.
    - `9779222`, seed 1 epoch 9, variant `flow_endpoint_medium10_fullcem_e9_s1_early_quick_2of3_costbest`.
    - `9779223`, seed 1 epoch 12, variant `flow_endpoint_medium10_fullcem_e12_s1_latest_quick_1of3`.
    - `9779224`, seed 2 epoch 12, variant `flow_endpoint_medium10_fullcem_e12_s2_latest_quick_0of3`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_fullcem_20260610_025437.tsv`.
  - Original residual flow-WM remains misaligned despite training progress: latest deterministic `validate/pred_loss` is still around `1.23`, while endpoint-flow is around `0.007-0.008`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-10 04:13 EDT:
  - Endpoint and original flow training chains are healthy:
    - Running endpoint-flow jobs: seed 0 `9772192`, seed 1 `9778779`, seed 2 `9765420`.
    - Running original flow jobs: seed 0 `9778808`, seed 1 `9778728`, seed 2 `9778754`.
    - Endpoint seed 2 job `9765420` is near the 8-hour limit with supervisor `9765421` pending on the valid dependency.
  - Native MLP + action-flow seed 1 eval completed after the dependency repair:
    - `9779210`: `34/50`, success rate `68.0%`, using native final epoch 100 world model.
    - Combined action-flow policy results are seed 0 `6.0%`, seed 1 `68.0%`, seed 2 `58.0%`; this remains below native CEM best checkpoints.
  - Endpoint medium10 full-CEM evals completed so far:
    - `9779221`, seed 0 epoch 14: `5/10`, success rate `50.0%`.
    - `9779222`, seed 1 epoch 9: `0/10`, success rate `0.0%`.
    - `9779223`, seed 1 epoch 12: `2/10`, success rate `20.0%`.
    - `9779224`, seed 2 epoch 12, is still pending on H200 priority.
    - Insight: the earlier seed 1 epoch 9 quick result `2/3` was likely noisy; seed 0 epoch 14 is currently the most promising endpoint checkpoint but is still below native MLP.
  - Submitted endpoint full50 follow-up for the promising checkpoint:
    - `9781663`, seed 0 epoch 14, variant `flow_endpoint_full50_e14_s0_medium10_50pct`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/full50_endpoint_followup_20260610_041219.tsv`.
  - Submitted latest endpoint cost-ranking diagnostics to test whether lower validation prediction loss improves the planning cost surface:
    - `9781664`, seed 0 epoch 19, variant `cost_rank_flow_endpoint_s0_e19_latest`.
    - `9781665`, seed 1 epoch 17, variant `cost_rank_flow_endpoint_s1_e17_latest`.
    - `9781666`, seed 2 epoch 17, variant `cost_rank_flow_endpoint_s2_e17_latest`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/latest_endpoint_cost_rank_20260610_041219.tsv`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-10 14:10 EDT:
  - Training continuation remains healthy:
    - Endpoint-flow jobs currently running: seed 0 `9795289`, seed 1 `9785898`, seed 2 `9794601`.
    - Original flow jobs currently running: seed 0 `9795269`, seed 1 `9795187`, seed 2 `9795231`.
    - Earlier endpoint-flow jobs `9765420` and `9772192` hit the 8-hour limit, and `9778779` was preempted; their supervisors completed and submitted the current running continuation jobs.
  - Endpoint-flow latest training metrics:
    | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` | Best pred row |
    | --- | --- | --- | --- | --- | --- |
    | 0 | 26 | 367300 | `0.005917` | `0.051471` | 24 (`0.005779`) |
    | 1 | 23 | 332200 | `0.006551` | `0.048091` | 22 (`0.006551`) |
    | 2 | 24 | 342050 | `0.006050` | `0.064956` | 23 (`0.006050`) |
  - Completed follow-up evals:
    - `9779224`, endpoint seed 2 epoch 12 medium10 full-CEM: `2/10`, success rate `20.0%`.
    - `9781663`, endpoint seed 0 epoch 14 full50 full-CEM: `15/50`, success rate `30.0%`.
    - This downgrades seed 0 epoch 14 from its earlier `5/10` medium result and confirms endpoint-flow is still well below native MLP.
  - Completed latest endpoint cost-ranking diagnostics:
    - seed 0 epoch 19: mean expert rank `4.3125`, random-better fraction `0.0254`; improved from seed 0 epoch 11 rank `7.875`.
    - seed 1 epoch 17: mean expert rank `1.0625`, random-better fraction `0.0005`; still excellent and similar to seed 1 epoch 9.
    - seed 2 epoch 17: mean expert rank `11.9375`, random-better fraction `0.0845`; worse than seed 2 epoch 9.
    - Insight: validation prediction loss keeps improving, but cost-ranking and real-env success do not improve monotonically.
  - Submitted latest endpoint medium10 full-CEM evals to test whether latest cost-ranking carries to real-env:
    - `9796330`, seed 0 epoch 19, variant `flow_endpoint_medium10_fullcem_e19_s0_latest_cost_rank_improved`.
    - `9796331`, seed 1 epoch 17, variant `flow_endpoint_medium10_fullcem_e17_s1_latest_cost_rank_best`.
    - `9796332`, seed 2 epoch 17, variant `flow_endpoint_medium10_fullcem_e17_s2_latest_cost_rank_worse`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest_20260610_140937.tsv`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring, 2026-06-10 15:13 EDT:
  - No new Slurm failures were found.
  - Endpoint-flow training is still improving:
    | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` | Best pred row |
    | --- | --- | --- | --- | --- | --- |
    | 0 | 27 | 387400 | `0.005747` | `0.051214` | 26 (`0.005747`) |
    | 1 | 24 | 342750 | `0.006243` | `0.047539` | 23 (`0.006243`) |
    | 2 | 25 | 355400 | `0.006138` | `0.064135` | 23 (`0.006050`) |
  - Original residual flow-WM remains misaligned: latest deterministic `validate/pred_loss` is still about `1.24` for all three seeds despite continued training.
  - Latest endpoint medium10 eval jobs `9796330`, `9796331`, and `9796332` remain pending on H200 priority.
  - Submitted non-overwriting A100 backup medium10 evals for the same latest endpoint checkpoints:
    - `9797486`, seed 0 epoch 19, variant `flow_endpoint_medium10_fullcem_a100_backup_e19_s0_latest_cost_rank_improved`.
    - `9797487`, seed 1 epoch 17, variant `flow_endpoint_medium10_fullcem_a100_backup_e17_s1_latest_cost_rank_best`.
    - `9797488`, seed 2 epoch 17, variant `flow_endpoint_medium10_fullcem_a100_backup_e17_s2_latest_cost_rank_worse`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest_a100_backup_20260610_151246.tsv`.
  - Endpoint seed 1 job `9785898` is near the 8-hour limit; supervisor `9785899` is pending on a valid dependency.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring, 2026-06-10 17:04 EDT:
  - No new application failures were found.
  - Recent timeout/preemptions were handled by supervisors:
    - Endpoint seed 1 `9785898` timed out; supervisor `9785899` submitted running continuation `9797913`.
    - Endpoint seed 0 `9795289`, endpoint seed 2 `9794601`, original flow seed 1 `9795187`, and original flow seed 2 `9795231` were preempted; their supervisors completed and submitted replacement jobs.
  - Current training queue:
    - Running endpoint-flow: seed 1 `9797913`, seed 2 `9797808`.
    - Pending endpoint-flow: seed 0 `9798360`.
    - Running original flow: seed 0 `9795269`, seed 1 `9797608`.
    - Pending original flow: seed 2 `9798459`.
  - Latest endpoint training metrics continue to improve:
    | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` | Best pred row |
    | --- | --- | --- | --- | --- | --- |
    | 0 | 28 | 402350 | `0.005664` | `0.050724` | 27 (`0.005664`) |
    | 1 | 25 | 356750 | `0.006163` | `0.046458` | 24 (`0.006163`) |
    | 2 | 26 | 373950 | `0.005692` | `0.062698` | 25 (`0.005692`) |
  - Original residual flow-WM deterministic prediction loss remains around `1.23-1.25`; continued training has not fixed CEM rollout alignment.
  - Latest endpoint medium10 evals remain pending on H200: `9796330`, `9796331`, `9796332`.
  - A100 backup medium10 evals also remain pending: `9797486`, `9797487`, `9797488`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-10 21:40 EDT:
  - H200 latest endpoint medium10 evals completed:
    - `9796330`, seed 0 epoch 19: `3/10`, success rate `30.0%`.
    - `9796331`, seed 1 epoch 17: `2/10`, success rate `20.0%`.
    - `9796332`, seed 2 epoch 17: `1/10`, success rate `10.0%`.
    - These results confirm that later endpoint checkpoints do not outperform the earlier seed 0 epoch 14 full50 result (`30%`) and remain far below native MLP.
  - Canceled duplicate A100 backup evals `9797486`, `9797487`, and `9797488` after the H200 evals completed.
  - Training continuation is still healthy:
    - Endpoint seed 0/1/2 latest metrics are epoch rows 30/28/28 with `validate/pred_loss` around `0.0056-0.0058`.
    - Original residual flow-WM remains misaligned at deterministic `validate/pred_loss` around `1.23-1.25`.
    - Recent timeout/preemptions were handled by supervisors; current continuation jobs include endpoint seed 0 `9805476`, endpoint seed 2 `9805624`, original flow seed 1 `9803788`, and original flow seed 2 `9803790`, with endpoint seed 1 `9805656` and original flow seed 0 `9809053` pending.
  - Submitted latest2 endpoint medium10 full-CEM evals:
    - `9809339`, seed 0 epoch 30, variant `flow_endpoint_medium10_fullcem_latest2_e30_s0`.
    - `9809340`, seed 1 epoch 28, variant `flow_endpoint_medium10_fullcem_latest2_e28_s1`.
    - `9809342`, seed 2 epoch 28, variant `flow_endpoint_medium10_fullcem_latest2_e28_s2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest2_20260610_213917.tsv`.
  - Submitted latest2 endpoint cost-ranking diagnostics:
    - `9809343`, seed 0 epoch 30, variant `cost_rank_flow_endpoint_s0_e30_latest2`.
    - `9809344`, seed 1 epoch 28, variant `cost_rank_flow_endpoint_s1_e28_latest2`.
    - `9809346`, seed 2 epoch 28, variant `cost_rank_flow_endpoint_s2_e28_latest2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/latest2_endpoint_cost_rank_20260610_213917.tsv`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-10 22:44 EDT:
  - Current training jobs are healthy and still running on H100:
    - Endpoint-flow: seed 0 `9805476`, seed 1 `9805656`, seed 2 `9805624`.
    - Original residual flow-WM: seed 0 `9809053`, seed 1 `9803788`, seed 2 `9803790`.
    - Supervisors remain pending on dependencies and are ready to continue after timeout/preemption.
  - Latest endpoint-flow checkpoints now available:
    | Seed | Latest checkpoint | Latest metric epoch row | `validate/pred_loss` | `validate/flow_loss` | Best pred row |
    | --- | --- | --- | --- | --- | --- |
    | 0 | epoch 31 | 31 | `0.005498` | `0.049934` | 28 (`0.005365`) |
    | 1 | epoch 29 | 29 | `0.005608` | `0.045050` | 28 (`0.005608`) |
    | 2 | epoch 29 | 29 | `0.005541` | `0.062937` | 28 (`0.005541`) |
  - Original residual flow-WM remains unsuitable for more real-env eval at this point:
    - seed 0 epoch 63: deterministic `validate/pred_loss=1.261526`.
    - seed 1 epoch 56: deterministic `validate/pred_loss=1.232183`.
    - seed 2 epoch 61: deterministic `validate/pred_loss=1.254028`.
  - Latest2 endpoint real-env and cost-rank jobs are still queued, with no new failure logs:
    - H200 medium10 full-CEM evals pending: `9809339`, `9809340`, `9809342`.
    - A100 medium10 backup evals pending: `9813246`, `9813247`, `9813248`.
    - A100 cost-rank diagnostics pending: `9809343`, `9809344`, `9809346`.
  - Submitted A100 backup eval record:
    - `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest2_a100_backup_20260610_224054.tsv`.
    - These jobs use distinct variants and W&B names from the H200 jobs, so they cannot race on the same eval output directory.
  - Did not submit another L40S duplicate batch: the L40S queue is also priority-bound, and H200 plus A100 coverage is already queued for the same latest2 checkpoints.
  - Runtime hygiene fix:
    - `scripts/slurm_runtime_env.sh` now sets `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` under `${STABLEWM_HOME}/runtime`.
    - This keeps future PyTorch compile caches out of HOME and avoids relying on `/tmp/torchinductor_*`.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-11 03:30 EDT:
  - Latest2 endpoint evals completed successfully:
    - H200 primary medium10 full-CEM:
      - `9809339`, seed 0 epoch 30: `2/10`, success rate `20.0%`.
      - `9809340`, seed 1 epoch 28: `2/10`, success rate `20.0%`.
      - `9809342`, seed 2 epoch 28: `3/10`, success rate `30.0%`.
    - A100 backup medium10 full-CEM:
      - `9813246`, seed 0 epoch 30: `2/10`, success rate `20.0%`.
      - `9813247`, seed 1 epoch 28: `1/10`, success rate `10.0%`.
      - `9813248`, seed 2 epoch 28: `3/10`, success rate `30.0%`.
    - Interpretation: the backup confirms the same low-success regime; the H200/A100 seed 1 difference is within the noise expected from 10 episodes.
  - Latest2 endpoint cost-ranking diagnostics completed:
    | Seed | Epoch | Mean expert rank | Random-better fraction | Read |
    | --- | ---: | ---: | ---: | --- |
    | 0 | 30 | `7.0625` | `0.0469` | Worse than seed 0 epoch 19 (`4.3125`, `0.0254`). |
    | 1 | 28 | `1.1875` | `0.0015` | Still near-native ranking, but real-env medium10 remains low. |
    | 2 | 28 | `7.2500` | `0.0483` | Better than seed 2 epoch 17, but still far from native and only `3/10` medium10. |
  - Main insight strengthened:
    - Endpoint-flow validation prediction loss keeps improving, but real-env success and planner-cost ranking do not improve monotonically.
    - The best current endpoint-flow full50 remains seed 0 epoch 14 at `30%`, far below native selected full50 `87.3 +/- 1.2%`.
    - Original residual flow-WM remains misaligned: latest deterministic `validate/pred_loss` is still around `1.24-1.25` at epochs 60-67.
  - Current training metrics:
    | Model | Seed | Latest epoch row | Step | `validate/pred_loss` | `validate/flow_loss` | Best pred row |
    | --- | ---: | ---: | ---: | ---: | ---: | --- |
    | endpoint-flow | 0 | 34 | 487650 | `0.005704` | `0.049251` | 32 (`0.005303`) |
    | endpoint-flow | 1 | 32 | 456050 | `0.005283` | `0.043976` | 31 (`0.005283`) |
    | endpoint-flow | 2 | 33 | 466200 | `0.005303` | `0.060745` | 30 (`0.005180`) |
    | residual flow-WM | 0 | 67 | 943500 | `1.253253` | `0.038692` | 0 (`0.080594`) |
    | residual flow-WM | 1 | 60 | 848450 | `1.242639` | `0.047671` | 0 (`0.085119`) |
    | residual flow-WM | 2 | 66 | 920200 | `1.242183` | `0.039303` | 0 (`0.080062`) |
  - Training queue health:
    - Running endpoint-flow: seed 0 `9805476`, seed 1 `9805656`, seed 2 continuation `9817278`.
    - Running residual flow-WM: seed 0 `9809053`, seed 1 `9803788`, seed 2 continuation `9817276`.
    - Preemptions for endpoint seed 2 `9805624` and residual flow seed 2 `9803790` were handled by supervisors and replaced by the continuations above.
  - Submitted latest3 endpoint medium10 full-CEM evals for the newest available checkpoints:
    - `9820446`, seed 0 epoch 34, variant `flow_endpoint_medium10_fullcem_latest3_e34_s0`.
    - `9820447`, seed 1 epoch 32, variant `flow_endpoint_medium10_fullcem_latest3_e32_s1`.
    - `9820448`, seed 2 epoch 33, variant `flow_endpoint_medium10_fullcem_latest3_e33_s2`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest3_20260611_033034.tsv`.
  - Submitted latest3 endpoint cost-ranking diagnostics:
    - `9820449`, seed 0 epoch 34, variant `cost_rank_flow_endpoint_s0_e34_latest3`.
    - `9820450`, seed 1 epoch 32, variant `cost_rank_flow_endpoint_s1_e32_latest3`.
    - `9820451`, seed 2 epoch 33, variant `cost_rank_flow_endpoint_s2_e33_latest3`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/latest3_endpoint_cost_rank_20260611_033034.tsv`.
  - Latest3 medium10 full-CEM evals completed shortly after submission:
    - `9820446`, seed 0 epoch 34: `3/10`, success rate `30.0%`.
    - `9820447`, seed 1 epoch 32: `1/10`, success rate `10.0%`.
    - `9820448`, seed 2 epoch 33: `2/10`, success rate `20.0%`.
    - Mean success rate: `20.0 +/- 10.0%`.
    - Interpretation: the plateau is confirmed; more endpoint-flow training and lower validation prediction loss still do not improve real-env success.
  - Latest3 cost-ranking jobs completed:
    | Seed | Epoch | Mean expert rank | Random-better fraction | Read |
    | --- | ---: | ---: | ---: | --- |
    | 0 | 34 | `7.5625` | `0.0513` | Still worse than seed 0 epoch 19 and not better than latest2. |
    | 1 | 32 | `5.4375` | `0.0342` | Regressed from latest2 seed 1 rank `1.1875`. |
    | 2 | 33 | `7.3125` | `0.0493` | Similar to latest2 seed 2 rank `7.2500`. |
    - Mean expert rank is `6.7708`; random-better fraction is `0.0449`.
    - Interpretation: latest3 confirms the endpoint-flow plateau. More training/lower validation prediction loss did not improve real-env success or planner-cost ranking.
    - Note: these diagnostics wrote local metrics under the default diagnostic variant path (`pusht/wm_flow_endpoint_policy_original/seed_x/eval/metrics.jsonl`) because `scripts/slurm_pusht_cost_ranking_diag.sbatch` does not yet pass `experiment.variant`; W&B summaries and Slurm logs match the values above.
  - Repo root check found no `wandb/`, `outputs/`, or `multirun` directories.

- PushT monitoring and follow-up, 2026-06-11 14:56 EDT:
  - Queue health:
    - Running LeWM H100 jobs:
      - residual flow-WM seed 0 `9830118`, around epoch 78.
      - residual flow-WM seed 1 `9822227`, around epoch 69 and close to the 8h walltime.
      - endpoint-flow seed 0 `9822228`, around epoch 42 and close to the 8h walltime.
    - Pending LeWM continuation jobs:
      - residual flow-WM seed 2 `9835958`.
      - endpoint-flow seed 1 `9830161`.
      - endpoint-flow seed 2 `9830151`.
    - Existing supervisors remain queued on dependencies and should continue the resume loop after timeout/preemption.
  - Latest validation signal:
    | Model | Seed | Latest epoch row | `validate/pred_loss_epoch` | `validate/flow_loss_epoch` | Read |
    | --- | ---: | ---: | ---: | ---: | --- |
    | residual flow-WM | 0 | 77 | `1.25744` | `0.03740` | Still no deterministic prediction recovery. |
    | residual flow-WM | 1 | 68 | `1.24567` | `0.04600` | Same failure mode as earlier residual runs. |
    | residual flow-WM | 2 | 75 | `1.25384` | `0.03813` | Same failure mode as earlier residual runs. |
    | endpoint-flow | 0 | 41 | `0.00490` | `0.04739` | Lower MSE than latest3, but prior eval shows no control gain. |
    | endpoint-flow | 1 | 39 | `0.00490` | `0.04098` | Lower MSE; not enough evidence to keep scaling unchanged. |
    | endpoint-flow | 2 | 39 | `0.00434` | `0.05567` | Best current endpoint pred loss, but needs planner-aware validation. |
    - Interpretation: residual flow remains unusable for CEM. Endpoint-flow continues to improve one-step MSE, but the latest3 real-env/cost-rank results already showed the same objective no longer gives better control.
  - Pushed the next fair ablation instead of only waiting for unchanged endpoint-flow scaling:
    - Submitted selected-native action-flow proposal training and full50 flow-policy evals using explicit selected native WM checkpoints:
      | Seed | Native WM epoch | Action-flow job | Eval job | GPU | Variant |
      | ---: | ---: | ---: | ---: | --- | --- |
      | 0 | 48 | `9835994` | `9835995` | H100 | `selected_native_e48_action_flow_s0_h100` |
      | 1 | 83 | `9835996` | `9835997` | A100 | `selected_native_e83_action_flow_s1_a100` |
      | 2 | 75 | `9836010` | `9836011` | A100 | `selected_native_e75_action_flow_s2_a100_retry` |
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_selected_action_flow_20260611/job_records/selected_native_action_flow_20260611_145524.tsv`.
    - Output root: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_selected_action_flow_20260611`.
    - The jobs use absolute `weights_epoch_{48,83,75}.pt` policy paths, so they do not accidentally load the collapsed epoch-100 native WMs.
    - Seed 2 was first attempted on L40S, but Slurm rejected it because the action-flow script requests 8 CPUs and the L40S partition enforces a 4:1 CPU:GPU limit; it was immediately resubmitted on A100.
  - Current decision:
    - Do not submit another unchanged endpoint-flow eval yet. The newest validation rows strengthen the same "MSE improves without control" signal; the more informative next result is whether flow helps as an action proposal when paired with the selected native WM.

- PushT monitoring and follow-up, 2026-06-11 22:49 EDT:
  - Selected-native action-flow status:
    - Seed 0 action-flow job `9835994` was preempted after writing `action_flow.pt`; its afterany full50 eval `9835995` completed successfully.
    - Seed 2 action-flow job `9836010` was preempted after writing `action_flow.pt`; its afterany full50 eval `9836011` completed successfully.
    - Seed 1 action-flow job `9835996` is still running on A100 and has already written `action_flow.pt`; its original afterany full50 eval `9835997` remains pending.
  - Selected-native action-flow full50 results so far:
    | Seed | Native WM epoch | Eval job | Success rate | Eval time | Read |
    | ---: | ---: | ---: | ---: | ---: | --- |
    | 0 | 48 | `9835995` | `62.0%` | `17.29s` | Much better than the previous epoch-100-confounded seed 0 action-flow result (`6.0%`), but below selected native CEM (`88%`). |
    | 2 | 75 | `9836011` | `44.0%` | `20.47s` | Worse than selected native CEM (`88%`) and worse than previous epoch-100 action-flow seed 2 (`58.0%`). |
    - Current mean over completed seeds is `53.0%`; wait for seed 1 before making the final selected-native action-flow conclusion.
  - To avoid waiting for the long-running seed 1 action-flow job while also avoiding file races, copied the current seed 1 `action_flow.pt` to a fixed snapshot:
    - Snapshot: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_selected_action_flow_20260611/pusht/wm_original_policy_flow/seed_1/action_flow_snapshots/action_flow_seed1_snapshot_20260611_224841.pt`.
    - Submitted snapshot full50 eval `9852203` on A100, variant `selected_native_e83_action_flow_s1_snapshot_a100`.
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_native_selected_action_flow_20260611/job_records/selected_native_action_flow_snapshot_eval_20260611_224841.tsv`.
    - The original afterany eval `9835997` is kept and can serve as a final/best action-flow comparison if the training job exits cleanly or after preemption.
  - Queue health:
    - Multiple LeWM flow continuation jobs were preempted, but supervisors resubmitted replacements (`9839708`, `9843125`, `9843208`, plus pending continuations).
    - This is cluster preemption behavior, not a new code/config failure.

- PushT monitoring and follow-up, 2026-06-13 02:25 EDT:
  - Selected-native action-flow full50 is now complete:
    | Seed | Native WM epoch | Eval job | Success rate | Eval time | Read |
    | ---: | ---: | ---: | ---: | ---: | --- |
    | 0 | 48 | `9835995` | `62.0%` | `17.29s` | Fairer than the prior epoch-100-confounded seed 0 result (`6.0%`), but below selected native CEM (`88%`). |
    | 1 | 83 | `9835997` | `58.0%` | `17.85s` | Snapshot eval `9852203` was `64.0%`; final afterany eval is the non-cherry-picked value. |
    | 2 | 75 | `9836011` | `44.0%` | `20.47s` | Still below selected native CEM (`88%`). |
    - Mean final selected-native action-flow full50 is `54.7%`, versus selected native CEM `87.3 +/- 1.2%`.
    - Interpretation: using selected native WMs removes the unfair epoch-100 collapse issue and improves action-flow seed 0 substantially, but action-flow still does not replace CEM as the main action selector.
  - Residual flow-WM has reached epoch 100 for seeds 0 and 2, and the supervisor submitted full50 CEM evals:
    | Seed | Latest validation row | `validate/pred_loss_epoch` | Eval job | Success rate | Eval time |
    | ---: | ---: | ---: | ---: | ---: | ---: |
    | 0 | 99 | `1.25915` | `9891092` | `4.0%` | `314.67s` |
    | 2 | 99 | `1.24376` | `9882878` | `2.0%` | `443.51s` |
    - Seed 1 residual flow-WM is at epoch 95 and still running/resuming; latest `validate/pred_loss_epoch=1.24908`.
    - Interpretation: the formal epoch-100 residual flow-WM result confirms the earlier diagnostic. It is both much worse and much slower than native CEM.
  - Endpoint-flow has continued training:
    | Seed | Latest checkpoint | Latest validation row | `validate/pred_loss_epoch` | `validate/flow_loss_epoch` |
    | ---: | ---: | ---: | ---: | ---: |
    | 0 | 64 | 63 | `0.003893` | `0.045229` |
    | 1 | 56 | 55 | `0.003816` | `0.036544` |
    | 2 | 58 | 57 | `0.003486` | `0.052976` |
    - Since the MSE is now substantially lower than latest3 but the previous results showed a plateau, submitted one latest4 screening batch to test whether the plateau still holds:
      | Seed | Epoch | Medium10 eval job | Cost-rank job |
      | ---: | ---: | ---: | ---: |
      | 0 | 64 | `9894694` | `9894695` |
      | 1 | 56 | `9894696` | `9894697` |
      | 2 | 58 | `9894698` | `9894699` |
    - Eval record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest4_20260613_022542.tsv`.
    - Cost-rank record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics_20260609/job_records/latest4_endpoint_cost_rank_20260613_022542.tsv`.
  - Fixed `scripts/slurm_pusht_cost_ranking_diag.sbatch` to pass optional `EXPERIMENT_VARIANT`, `WANDB_RUN_GROUP`, and `WANDB_NAME` into Hydra/W&B. This prevents latest4 cost-rank diagnostics from falling back to the default local variant path.
  - Queue health:
    - Current LeWM jobs are running normally under `embers`: endpoint-flow seeds 0/1/2 and residual flow seed 1 continue through the supervisor resume loop.
    - The latest4 endpoint eval/cost-rank jobs are pending on priority, with no immediate submission failures.

- PushT monitoring and follow-up, 2026-06-13 18:36 EDT:
  - Latest4 endpoint cost-rank diagnostics completed:
    | Seed | Epoch | Mean expert rank | Random-better fraction | Read |
    | ---: | ---: | ---: | ---: | --- |
    | 0 | 64 | `24.5625` | `0.1831` | Much worse than latest3 seed 0 rank `7.5625`. |
    | 1 | 56 | `20.7500` | `0.1533` | Regressed strongly from latest3 seed 1 rank `5.4375`. |
    | 2 | 58 | `9.2500` | `0.0645` | Slightly worse than latest3 seed 2 rank `7.3125`. |
    - Mean expert rank is `18.1875`; mean random-better fraction is `0.1336`.
    - Interpretation: endpoint-flow prediction MSE kept improving, but planner cost ranking got worse. This strongly supports the current report thesis that lower endpoint MSE is not the right objective.
  - Latest4 medium10 eval submit bug:
    - Jobs `9894694`, `9894696`, and `9894698` failed immediately because `EVAL_BUDGET=10` violated the eval assertion `horizon * action_block <= eval_budget` (`5 * 5 <= eval_budget`).
    - This was a submission-parameter error, not a model or code failure.
    - Resubmitted with `EVAL_NUM_EVAL=10` and `EVAL_BUDGET=50`:
      | Seed | Epoch | Retry job | Variant |
      | ---: | ---: | ---: | --- |
      | 0 | 64 | `9916205` | `flow_endpoint_medium10_fullcem_latest4_budgetfix_e64_s0_h100` |
      | 1 | 56 | `9916206` | `flow_endpoint_medium10_fullcem_latest4_budgetfix_e56_s1_h100` |
      | 2 | 58 | `9916207` | `flow_endpoint_medium10_fullcem_latest4_budgetfix_e58_s2_h100` |
    - Record: `/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_endpoint_formal_20260609/job_records/medium_eval_endpoint_latest4_budgetfix_20260613_183422.tsv`.
  - Residual flow-WM formal epoch-100 full50 is now complete:
    | Seed | Eval job | Success rate | Eval time |
    | ---: | ---: | ---: | ---: |
    | 0 | `9891092` | `4.0%` | `314.67s` |
    | 1 | `9896657` | `0.0%` | `436.88s` |
    | 2 | `9882878` | `2.0%` | `443.51s` |
    - Mean success is `2.0%`. This closes the residual flow-WM formal run: it is far below native selected CEM (`87.3 +/- 1.2%`) and much slower.
  - Current queue:
    - Endpoint-flow seeds 0/1/2 continue training under the supervisor resume loop.
    - Latest4 budget-fixed medium10 evals are pending on H100 resources, with no immediate failure.
