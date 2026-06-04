#!/bin/bash
set -euo pipefail

TASK="${1:?usage: scripts/submit_flow_2x2_wm.sh <pusht|cube>}"

case "${TASK}" in
  pusht)
    DATA=pusht
    ;;
  cube)
    DATA=ogb
    ;;
  *)
    echo "Unknown task: ${TASK}" >&2
    exit 2
    ;;
esac

for SEED in 0 1; do
  sbatch --parsable \
    --job-name="lewm-${TASK}-orig-s${SEED}" \
    --export=ALL,TASK="${TASK}",DATA="${DATA}",WM_VARIANT=original,MODEL=lewm,OUTPUT_MODEL_NAME=lewm,SEED="${SEED}" \
    scripts/slurm_flow_2x2_train.sbatch

  sbatch --parsable \
    --job-name="lewm-${TASK}-flow-s${SEED}" \
    --export=ALL,TASK="${TASK}",DATA="${DATA}",WM_VARIANT=flow,MODEL=lewm_flow,OUTPUT_MODEL_NAME=lewm_flow,SEED="${SEED}" \
    scripts/slurm_flow_2x2_train.sbatch
done
