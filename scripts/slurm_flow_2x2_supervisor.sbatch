#!/bin/bash
#SBATCH --job-name=lewm-supervisor
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${STABLEWM_HOME}/experiments/flow_2x2_20260604
source "${REPO}/scripts/slurm_runtime_env.sh"

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${LEWM_EXPERIMENT_ROOT}/job_records"
cd "${REPO}"

: "${TASK:?TASK is required: pusht or cube}"
: "${DATA:?DATA is required: pusht or ogb}"
: "${WM_VARIANT:?WM_VARIANT is required: original or flow}"
: "${MODEL:?MODEL is required: lewm or lewm_flow}"
: "${OUTPUT_MODEL_NAME:?OUTPUT_MODEL_NAME is required}"
: "${SEED:?SEED is required}"

WM_EPOCH="${WM_EPOCH:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-${WM_EPOCH}}"
ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
WANDB_MODE="${WANDB_MODE:-online}"
UPSTREAM_JOB_ID="${UPSTREAM_JOB_ID:-}"
RECORD="${RECORD:-${LEWM_EXPERIMENT_ROOT}/job_records/supervisor_$(date +%Y%m%d).tsv}"

if [ "${TASK}" = "pusht" ]; then
  CONFIG=pusht
else
  CONFIG=cube
fi

WORLD_MODEL="${TASK}/wm_${WM_VARIANT}_policy_original/seed_${SEED}/${MODEL}/weights_epoch_${WM_EPOCH}.pt"
TARGET_CKPT="${LEWM_EXPERIMENT_ROOT}/checkpoints/${WORLD_MODEL}"
ACTION_MODEL_PATH="${LEWM_EXPERIMENT_ROOT}/${TASK}/wm_${WM_VARIANT}_policy_flow/seed_${SEED}/action_flow/action_flow.pt"
DONE_MARKER="${LEWM_EXPERIMENT_ROOT}/job_records/downstream_${TASK}_${WM_VARIANT}_seed${SEED}_epoch${WM_EPOCH}.done"

if [ ! -s "${RECORD}" ]; then
  printf 'time\tphase\ttask\twm_variant\tpolicy_variant\tseed\tjob_id\tdependency\tnote\n' > "${RECORD}"
fi

log_record() {
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$1" "${TASK}" "${WM_VARIANT}" "$2" "${SEED}" "$3" "$4" "$5" | tee -a "${RECORD}"
}

upstream_state=""
if [ -n "${UPSTREAM_JOB_ID}" ]; then
  upstream_state=$(sacct -j "${UPSTREAM_JOB_ID}" --format=State -X -n | awk 'NF {print $1; exit}' || true)
  echo "Upstream job ${UPSTREAM_JOB_ID} state: ${upstream_state:-unknown}"
fi

case "${upstream_state}" in
  FAILED|OUT_OF_MEMORY|CANCELLED|CANCELLED+)
    log_record "supervisor_stop" "original" "${SLURM_JOB_ID}" "afterany:${UPSTREAM_JOB_ID}" "upstream_state=${upstream_state}"
    exit 1
    ;;
esac

echo "Target checkpoint: ${TARGET_CKPT}"

if [ -f "${TARGET_CKPT}" ]; then
  if [ -f "${DONE_MARKER}" ]; then
    echo "Downstream already submitted: ${DONE_MARKER}"
    log_record "supervisor_done" "original" "${SLURM_JOB_ID}" "" "marker_exists"
    exit 0
  fi

  echo "Final world model checkpoint exists; submitting downstream jobs."
  af_id=$(sbatch --parsable \
    --job-name="af-${TASK}-${WM_VARIANT}-s${SEED}" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",TASK="${TASK}",DATA="${DATA}",WM_VARIANT="${WM_VARIANT}",WORLD_MODEL="${WORLD_MODEL}",SEED="${SEED}" \
    scripts/slurm_flow_2x2_action_flow.sbatch)
  log_record "action_flow" "flow" "${af_id}" "" "checkpoint_ready"

  cem_id=$(sbatch --parsable \
    --job-name="ev-${TASK}-${WM_VARIANT}-cem-s${SEED}" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",TASK="${TASK}",CONFIG_NAME="${CONFIG}",WM_VARIANT="${WM_VARIANT}",POLICY_VARIANT=original,POLICY="${WORLD_MODEL}",SEED="${SEED}" \
    scripts/slurm_flow_2x2_eval.sbatch)
  log_record "eval" "original" "${cem_id}" "" "checkpoint_ready"

  flow_id=$(sbatch --parsable \
    --dependency=afterok:${af_id} \
    --job-name="ev-${TASK}-${WM_VARIANT}-flow-s${SEED}" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",TASK="${TASK}",CONFIG_NAME="${CONFIG}",WM_VARIANT="${WM_VARIANT}",POLICY_VARIANT=flow,POLICY="${WORLD_MODEL}",ACTION_MODEL_PATH="${ACTION_MODEL_PATH}",SEED="${SEED}" \
    scripts/slurm_flow_2x2_eval.sbatch)
  log_record "eval" "flow" "${flow_id}" "afterok:${af_id}" "checkpoint_ready"

  printf 'submitted_at=%s\naction_flow=%s\ncem_eval=%s\nflow_eval=%s\n' \
    "$(date --iso-8601=seconds)" "${af_id}" "${cem_id}" "${flow_id}" > "${DONE_MARKER}"
else
  echo "Final checkpoint is not ready; submitting the next resumable world-model chunk."
  train_id=$(sbatch --parsable \
    --job-name="lewm-${TASK}-${WM_VARIANT}-s${SEED}" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",MAX_EPOCHS="${MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",RESUME_AUTO=True,TASK="${TASK}",DATA="${DATA}",WM_VARIANT="${WM_VARIANT}",MODEL="${MODEL}",OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME}",SEED="${SEED}" \
    scripts/slurm_flow_2x2_train.sbatch)
  log_record "world_model" "original" "${train_id}" "" "resume_until_epoch_${WM_EPOCH}"

  supervisor_id=$(sbatch --parsable \
    --dependency=afterany:${train_id} \
    --job-name="sup-${TASK}-${WM_VARIANT}-s${SEED}" \
    --export=ALL,WANDB_MODE="${WANDB_MODE}",WM_EPOCH="${WM_EPOCH}",MAX_EPOCHS="${MAX_EPOCHS}",ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",TASK="${TASK}",DATA="${DATA}",WM_VARIANT="${WM_VARIANT}",MODEL="${MODEL}",OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME}",SEED="${SEED}",UPSTREAM_JOB_ID="${train_id}",RECORD="${RECORD}" \
    scripts/slurm_flow_2x2_supervisor.sbatch)
  log_record "supervisor" "original" "${supervisor_id}" "afterany:${train_id}" "watch_resume_chunk"
fi
