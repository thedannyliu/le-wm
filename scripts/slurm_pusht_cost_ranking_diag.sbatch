#!/bin/bash
#SBATCH --job-name=pusht-cost-rank
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_cost_diagnostics/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${LEWM_EXPERIMENT_ROOT:-${STABLEWM_HOME}/experiments/pusht_cost_diagnostics}
source "${REPO}/scripts/slurm_runtime_env.sh"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-lewm-flow-2x2}

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${LEWM_JOB_WORKDIR}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${LEWM_JOB_WORKDIR}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

: "${WM_VARIANT:?WM_VARIANT is required}"
: "${POLICY_VARIANT:?POLICY_VARIANT is required}"
: "${POLICY:?POLICY is required}"
: "${SEED:?SEED is required}"

EXTRA_ARGS=()
if [ -n "${EXPERIMENT_VARIANT:-}" ]; then
  EXTRA_ARGS+=(experiment.variant="${EXPERIMENT_VARIANT}")
fi
if [ -n "${WANDB_RUN_GROUP:-}" ]; then
  EXTRA_ARGS+=(wandb.config.group="${WANDB_RUN_GROUP}")
fi
if [ -n "${WANDB_NAME:-}" ]; then
  EXTRA_ARGS+=(wandb.config.name="${WANDB_NAME}")
fi

"${PY}" "${REPO}/diagnose_cost_ranking.py" \
  --config-name=pusht \
  seed="${SEED}" \
  policy="${POLICY}" \
  cache_dir="${STABLEWM_HOME}" \
  checkpoint_cache_dir="${CHECKPOINT_ROOT:-${LEWM_EXPERIMENT_ROOT}}" \
  experiment.output_root="${LEWM_EXPERIMENT_ROOT}" \
  experiment.task=pusht \
  experiment.wm_variant="${WM_VARIANT}" \
  experiment.policy_variant="${POLICY_VARIANT}" \
  world.max_episode_steps=100 \
  eval.num_eval="${DIAG_NUM_EVAL:-16}" \
  eval.eval_budget=50 \
  +diagnostic.num_eval="${DIAG_NUM_EVAL:-16}" \
  +diagnostic.num_random="${DIAG_NUM_RANDOM:-128}" \
  hydra.run.dir="${LEWM_RUNTIME_ROOT}/hydra/${SLURM_JOB_ID}/pusht_${WM_VARIANT}_${POLICY_VARIANT}_cost_rank_seed${SEED}" \
  wandb.enabled=True \
  "${EXTRA_ARGS[@]}"
