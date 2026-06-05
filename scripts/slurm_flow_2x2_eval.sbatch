#!/bin/bash
#SBATCH --job-name=lewm-eval
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
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

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${LEWM_JOB_WORKDIR}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${LEWM_JOB_WORKDIR}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

: "${TASK:?TASK is required}"
: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${WM_VARIANT:?WM_VARIANT is required}"
: "${POLICY_VARIANT:?POLICY_VARIANT is required}"
: "${POLICY:?POLICY is required}"
: "${SEED:?SEED is required}"

COMMON_ARGS=(
  --config-name="${CONFIG_NAME}"
  seed="${SEED}"
  policy="${POLICY}"
  cache_dir="${STABLEWM_HOME}"
  checkpoint_cache_dir="${LEWM_EXPERIMENT_ROOT}"
  experiment.output_root="${LEWM_EXPERIMENT_ROOT}"
  experiment.task="${TASK}"
  experiment.wm_variant="${WM_VARIANT}"
  experiment.policy_variant="${POLICY_VARIANT}"
  world.max_episode_steps=100
  hydra.run.dir="${LEWM_RUNTIME_ROOT}/hydra/${SLURM_JOB_ID}/${TASK}_${WM_VARIANT}_${POLICY_VARIANT}_seed${SEED}_eval"
  wandb.enabled=True
)

if [ "${TASK}" = "cube" ]; then
  COMMON_ARGS+=(eval.dataset_name="${STABLEWM_HOME}/ogbench/cube_single_expert.h5")
fi

if [ "${POLICY_VARIANT}" = "flow" ]; then
  : "${ACTION_MODEL_PATH:?ACTION_MODEL_PATH is required for flow policy eval}"
  "${PY}" "${REPO}/eval.py" "${COMMON_ARGS[@]}" \
    solver=flow \
    solver.action_model_path="${ACTION_MODEL_PATH}" \
    solver.action_model_cache_dir=/
else
  "${PY}" "${REPO}/eval.py" "${COMMON_ARGS[@]}" solver=cem
fi
