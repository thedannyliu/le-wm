#!/bin/bash
#SBATCH --job-name=lewm-pusht-lance
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/pusht_h100_speed_20260605/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export LEWM_EXPERIMENT_ROOT=${STABLEWM_HOME}/experiments/pusht_h100_speed_20260605
source "${REPO}/scripts/slurm_runtime_env.sh"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LEWM_EXPERIMENT_ROOT}/slurm" "${STABLEWM_HOME}/datasets"
cd "${LEWM_JOB_WORKDIR}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"
SRC="${STABLEWM_HOME}/datasets/pusht_expert_train.h5"
ALT_SRC="${STABLEWM_HOME}/pusht_expert_train.h5"
DEST="${STABLEWM_HOME}/datasets/pusht_expert_train.lance"

if [ ! -f "${SRC}" ] && [ -f "${ALT_SRC}" ]; then
  SRC="${ALT_SRC}"
fi

if [ -d "${DEST}" ]; then
  echo "Lance dataset already exists: ${DEST}"
  exit 0
fi

echo "Converting ${SRC} -> ${DEST}"
"${PY}" - <<PY
from stable_worldmodel.data import convert
convert("${SRC}", "${DEST}", source_format="hdf5", dest_format="lance")
PY

echo "Done: ${DEST}"
