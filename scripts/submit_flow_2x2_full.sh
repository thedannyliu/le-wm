#!/bin/bash
set -euo pipefail

ROOT=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604
mkdir -p "${ROOT}/job_records"

WM_EPOCH="${WM_EPOCH:-100}"
MAX_EPOCHS="${MAX_EPOCHS:-${WM_EPOCH}}"
ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RECORD="${ROOT}/job_records/resubmit_$(date +%Y%m%d_%H%M%S).tsv"

printf 'phase\ttask\twm_variant\tpolicy_variant\tseed\tjob_id\tdependency\n' > "${RECORD}"

declare -A TRAIN_IDS

for TASK in pusht cube; do
  if [ "${TASK}" = pusht ]; then
    DATA=pusht
  else
    DATA=ogb
  fi

  for SEED in 0 1; do
    jid=$(sbatch --parsable \
      --job-name="lewm-${TASK}-orig-s${SEED}" \
      --export=ALL,WANDB_MODE=online,MAX_EPOCHS="${MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",RESUME_AUTO=True,TASK="${TASK}",DATA="${DATA}",WM_VARIANT=original,MODEL=lewm,OUTPUT_MODEL_NAME=lewm,SEED="${SEED}" \
      scripts/slurm_flow_2x2_train.sbatch)
    TRAIN_IDS["${TASK} original ${SEED}"]=${jid}
    printf 'world_model\t%s\toriginal\toriginal\t%s\t%s\t\n' "${TASK}" "${SEED}" "${jid}" | tee -a "${RECORD}"

    jid=$(sbatch --parsable \
      --job-name="lewm-${TASK}-flow-s${SEED}" \
      --export=ALL,WANDB_MODE=online,MAX_EPOCHS="${MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",RESUME_AUTO=True,TASK="${TASK}",DATA="${DATA}",WM_VARIANT=flow,MODEL=lewm_flow,OUTPUT_MODEL_NAME=lewm_flow,SEED="${SEED}" \
      scripts/slurm_flow_2x2_train.sbatch)
    TRAIN_IDS["${TASK} flow ${SEED}"]=${jid}
    printf 'world_model\t%s\tflow\toriginal\t%s\t%s\t\n' "${TASK}" "${SEED}" "${jid}" | tee -a "${RECORD}"
  done
done

for TASK in pusht cube; do
  if [ "${TASK}" = pusht ]; then
    DATA=pusht
    CONFIG=pusht
  else
    DATA=ogb
    CONFIG=cube
  fi

  for WM in original flow; do
    for SEED in 0 1; do
      train_id=${TRAIN_IDS["${TASK} ${WM} ${SEED}"]}
      if [ "${WM}" = original ]; then
        MODEL_NAME=lewm
      else
        MODEL_NAME=lewm_flow
      fi
      supervisor_id=$(sbatch --parsable \
        --dependency=afterany:${train_id} \
        --job-name="sup-${TASK}-${WM}-s${SEED}" \
        --export=ALL,WANDB_MODE=online,WM_EPOCH="${WM_EPOCH}",MAX_EPOCHS="${MAX_EPOCHS}",ACTION_MAX_EPOCHS="${ACTION_MAX_EPOCHS}",NUM_WORKERS="${NUM_WORKERS}",TASK="${TASK}",DATA="${DATA}",WM_VARIANT="${WM}",MODEL="${MODEL_NAME}",OUTPUT_MODEL_NAME="${MODEL_NAME}",SEED="${SEED}",UPSTREAM_JOB_ID="${train_id}",RECORD="${RECORD}" \
        scripts/slurm_flow_2x2_supervisor.sbatch)
      printf 'supervisor\t%s\t%s\toriginal\t%s\t%s\tafterany:%s\n' "${TASK}" "${WM}" "${SEED}" "${supervisor_id}" "${train_id}" | tee -a "${RECORD}"
    done
  done
done

echo "RECORD=${RECORD}"
