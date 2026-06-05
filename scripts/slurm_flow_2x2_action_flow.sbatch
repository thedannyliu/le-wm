#!/bin/bash
#SBATCH --job-name=lewm-action-flow
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${STABLEWM_HOME}/experiments/flow_2x2_20260604
source "${REPO}/scripts/slurm_runtime_env.sh"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-lewm-flow-2x2}
export ACTION_MAX_EPOCHS=${ACTION_MAX_EPOCHS:-20}
export NUM_WORKERS=${NUM_WORKERS:-6}

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${LEWM_JOB_WORKDIR}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${LEWM_JOB_WORKDIR}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

: "${TASK:?TASK is required}"
: "${DATA:?DATA is required}"
: "${WM_VARIANT:?WM_VARIANT is required}"
: "${WORLD_MODEL:?WORLD_MODEL is required}"
: "${SEED:?SEED is required}"

echo "Action flow: Task=${TASK} Data=${DATA} WM=${WM_VARIANT} Seed=${SEED}"
echo "World model: ${WORLD_MODEL}"
echo "ACTION_MAX_EPOCHS=${ACTION_MAX_EPOCHS} NUM_WORKERS=${NUM_WORKERS}"
echo "Runtime root=${LEWM_RUNTIME_ROOT}"

EXTRA_ARGS=()
if [ "${TASK}" = "cube" ]; then
  EXTRA_ARGS+=(data.dataset.name="${STABLEWM_HOME}/ogbench/cube_single_expert.h5")
fi

"${PY}" "${REPO}/train_action_flow.py" \
  data="${DATA}" \
  seed="${SEED}" \
  cache_dir="${STABLEWM_HOME}" \
  checkpoint_cache_dir="${LEWM_EXPERIMENT_ROOT}" \
  experiment.output_root="${LEWM_EXPERIMENT_ROOT}" \
  experiment.task="${TASK}" \
  experiment.wm_variant="${WM_VARIANT}" \
  experiment.policy_variant=flow \
  world_model="${WORLD_MODEL}" \
  device=cuda \
  trainer.max_epochs="${ACTION_MAX_EPOCHS}" \
  loader.num_workers="${NUM_WORKERS}" \
  loader.pin_memory=True \
  loader.persistent_workers=True \
  loader.prefetch_factor=3 \
  hydra.run.dir="${LEWM_RUNTIME_ROOT}/hydra/${SLURM_JOB_ID}/${TASK}_${WM_VARIANT}_seed${SEED}_action_flow" \
  wandb.enabled=True \
  "${EXTRA_ARGS[@]}"
