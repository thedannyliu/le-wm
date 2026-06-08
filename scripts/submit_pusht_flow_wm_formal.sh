#!/bin/bash
set -euo pipefail

ROOT="${ROOT:-/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_flow_wm_formal_20260608}"
mkdir -p "${ROOT}/slurm" "${ROOT}/job_records"

DEPENDENCY="${DEPENDENCY:-}"
WM_EPOCH="${WM_EPOCH:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-${WM_EPOCH}}"
ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS:-20}"
WANDB_MODE="${WANDB_MODE:-online}"
PIN_MEMORY="${PIN_MEMORY:-False}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-False}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
RECORD="${ROOT}/job_records/submit_$(date +%Y%m%d_%H%M%S).tsv"

printf 'time\tphase\ttask\twm_variant\tpolicy_variant\tseed\tgpu\tjob_id\tdependency\tnote\n' > "${RECORD}"

submit_seed() {
  local seed="$1"
  local gpu_label="$2"
  local partition="$3"
  local gres="$4"
  local cpus="$5"
  local mem="$6"
  local workers="$7"
  local subdir="pusht_flow_wm_formal_20260608_${gpu_label}_seed${seed}"
  local dep_args=()

  if [ -n "${DEPENDENCY}" ]; then
    dep_args=(--dependency="${DEPENDENCY}")
  fi

  local train_id
  train_id=$(sbatch --parsable \
    "${dep_args[@]}" \
    --partition="${partition}" \
    --gres="${gres}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --job-name="lewm-pusht-flowwm-${gpu_label}-s${seed}" \
    --output="${ROOT}/slurm/%x-%j.out" \
    --error="${ROOT}/slurm/%x-%j.err" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",LEWM_EXPERIMENT_ROOT="${ROOT}",MAX_EPOCHS="${MAX_EPOCHS}",NUM_WORKERS="${workers}",PIN_MEMORY="${PIN_MEMORY}",PERSISTENT_WORKERS="${PERSISTENT_WORKERS}",PREFETCH_FACTOR="${PREFETCH_FACTOR}",RESUME_AUTO=True,SUBDIR="${subdir}",TASK=pusht,DATA=pusht,WM_VARIANT=flow,MODEL=lewm_flow,OUTPUT_MODEL_NAME=lewm_flow,SEED="${seed}" \
    scripts/slurm_flow_2x2_train.sbatch)

  printf '%s\tworld_model\tpusht\tflow\toriginal\t%s\t%s\t%s\t%s\tformal_residual_flow_wm_epoch_%s\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${gpu_label}" "${train_id}" "${DEPENDENCY}" "${WM_EPOCH}" | tee -a "${RECORD}"

  local supervisor_id
  supervisor_id=$(sbatch --parsable \
    --dependency=afterany:"${train_id}" \
    --job-name="sup-pusht-flowwm-${gpu_label}-s${seed}" \
    --output="${ROOT}/slurm/%x-%j.out" \
    --error="${ROOT}/slurm/%x-%j.err" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",LEWM_EXPERIMENT_ROOT="${ROOT}",WM_EPOCH="${WM_EPOCH}",MAX_EPOCHS="${MAX_EPOCHS}",ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS}",NUM_WORKERS="${workers}",PIN_MEMORY="${PIN_MEMORY}",PERSISTENT_WORKERS="${PERSISTENT_WORKERS}",PREFETCH_FACTOR="${PREFETCH_FACTOR}",SUBDIR="${subdir}",TASK=pusht,DATA=pusht,WM_VARIANT=flow,MODEL=lewm_flow,OUTPUT_MODEL_NAME=lewm_flow,SEED="${seed}",SUBMIT_ACTION_FLOW=False,SUBMIT_FLOW_EVAL=False,SUBMIT_CEM_EVAL=True,UPSTREAM_JOB_ID="${train_id}",RECORD="${RECORD}" \
    scripts/slurm_flow_2x2_supervisor.sbatch)

  printf '%s\tsupervisor\tpusht\tflow\toriginal\t%s\t%s\t%s\tafterany:%s\twatch_resume_chunk_then_cem_eval\n' \
    "$(date --iso-8601=seconds)" "${seed}" "${gpu_label}" "${supervisor_id}" "${train_id}" | tee -a "${RECORD}"
}

submit_seed 0 h100 gpu-h100 gpu:h100:1 8 160G 6
submit_seed 1 a100 gpu-a100 gpu:a100:1 8 160G 6
submit_seed 2 h100 gpu-h100 gpu:h100:1 8 160G 6

echo "RECORD=${RECORD}"
