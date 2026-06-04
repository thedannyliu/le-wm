#!/bin/bash
#SBATCH --job-name=lewm-action-flow
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${STABLEWM_HOME}/experiments/flow_2x2_20260604
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-lewm-flow-2x2}

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm"
cd "${REPO}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

: "${TASK:?TASK is required}"
: "${DATA:?DATA is required}"
: "${WM_VARIANT:?WM_VARIANT is required}"
: "${WORLD_MODEL:?WORLD_MODEL is required}"
: "${SEED:?SEED is required}"

echo "Action flow: Task=${TASK} Data=${DATA} WM=${WM_VARIANT} Seed=${SEED}"
echo "World model: ${WORLD_MODEL}"

EXTRA_ARGS=()
if [ "${TASK}" = "cube" ]; then
  EXTRA_ARGS+=(data.dataset.name="${STABLEWM_HOME}/ogbench/cube_single_expert.h5")
fi

"${PY}" train_action_flow.py \
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
  wandb.enabled=True \
  "${EXTRA_ARGS[@]}"
