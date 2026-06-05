#!/bin/bash
#SBATCH --job-name=lewm-pusht-h100-speed
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${LEWM_EXPERIMENT_ROOT:-${STABLEWM_HOME}/experiments/pusht_h100_speed_20260605}
source "${REPO}/scripts/slurm_runtime_env.sh"
export PYTHONNOUSERSITE=1
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-lewm-flow-2x2}
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${LEWM_JOB_WORKDIR}"
cd "${LEWM_JOB_WORKDIR}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"
MAX_EPOCHS=${MAX_EPOCHS:-1}
SEED=${SEED:-0}
GPU_LABEL=${GPU_LABEL:-h100}
NUM_WORKERS=${NUM_WORKERS:-6}

python_info=$("${PY}" - <<'PY'
import torch
print(torch.__version__, "cuda_available=", torch.cuda.is_available(), "device=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
)
echo "Python/Torch: ${python_info}"
echo "Task=pusht Data=pusht WM=original Model=lewm Seed=${SEED} GPU_LABEL=${GPU_LABEL}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "Runtime root=${LEWM_RUNTIME_ROOT}"

"${PY}" "${REPO}/train.py" \
  data=pusht \
  model=lewm \
  seed="${SEED}" \
  cache_dir="${STABLEWM_HOME}" \
  experiment.output_root="${LEWM_EXPERIMENT_ROOT}" \
  experiment.task=pusht \
  experiment.wm_variant=original \
  experiment.policy_variant=original \
  output_model_name=lewm \
  subdir="pusht_original_${GPU_LABEL}_speed_seed${SEED}" \
  +checkpoint_run_name="pusht/${GPU_LABEL}_speed/seed_${SEED}/lewm" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.precision=bf16 \
  trainer.max_epochs="${MAX_EPOCHS}" \
  loader.num_workers="${NUM_WORKERS}" \
  loader.pin_memory=True \
  loader.persistent_workers=True \
  loader.prefetch_factor=3 \
  hydra.run.dir="${LEWM_RUNTIME_ROOT}/hydra/${SLURM_JOB_ID}/pusht_${GPU_LABEL}_speed_seed${SEED}" \
  wandb.enabled=True \
  wandb.config.group="pusht/${GPU_LABEL}_speed/original" \
  wandb.config.name="pusht_original_${GPU_LABEL}_speed_seed${SEED}" \
  wandb.config.id="pusht_original_${GPU_LABEL}_speed_seed${SEED}"
