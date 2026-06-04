#!/bin/bash
#SBATCH --job-name=lewm-flow-train
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
export MAX_EPOCHS=${MAX_EPOCHS:-100}
export NUM_WORKERS=${NUM_WORKERS:-2}
export RESUME_AUTO=${RESUME_AUTO:-True}

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm"
cd "${REPO}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

: "${TASK:?TASK is required: pusht or cube}"
: "${DATA:?DATA is required: pusht or ogb}"
: "${WM_VARIANT:?WM_VARIANT is required: original or flow}"
: "${MODEL:?MODEL is required: lewm or lewm_flow}"
: "${OUTPUT_MODEL_NAME:?OUTPUT_MODEL_NAME is required}"
: "${SEED:?SEED is required}"

python_info=$("${PY}" - <<'PY'
import torch
print(torch.__version__, "cuda_available=", torch.cuda.is_available())
PY
)
echo "Python/Torch: ${python_info}"
echo "Task=${TASK} Data=${DATA} WM=${WM_VARIANT} Model=${MODEL} Seed=${SEED}"
echo "WANDB_MODE=${WANDB_MODE}"
echo "MAX_EPOCHS=${MAX_EPOCHS} NUM_WORKERS=${NUM_WORKERS} RESUME_AUTO=${RESUME_AUTO}"
echo "Runtime root=${LEWM_RUNTIME_ROOT}"

EXTRA_ARGS=()
if [ "${TASK}" = "cube" ]; then
  EXTRA_ARGS+=(data.dataset.name="${STABLEWM_HOME}/ogbench/cube_single_expert.h5")
fi

"${PY}" train.py \
  data="${DATA}" \
  model="${MODEL}" \
  seed="${SEED}" \
  cache_dir="${STABLEWM_HOME}" \
  experiment.output_root="${LEWM_EXPERIMENT_ROOT}" \
  experiment.task="${TASK}" \
  experiment.wm_variant="${WM_VARIANT}" \
  experiment.policy_variant=original \
  output_model_name="${OUTPUT_MODEL_NAME}" \
  subdir="${TASK}_${WM_VARIANT}_seed${SEED}" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16 \
  trainer.max_epochs="${MAX_EPOCHS}" \
  resume.auto="${RESUME_AUTO}" \
  loader.num_workers="${NUM_WORKERS}" \
  loader.pin_memory=False \
  loader.persistent_workers=False \
  loader.prefetch_factor=1 \
  wandb.enabled=True \
  "${EXTRA_ARGS[@]}"
