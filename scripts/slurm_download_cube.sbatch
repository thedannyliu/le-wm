#!/bin/bash
#SBATCH --job-name=lewm-cube-data
#SBATCH --account=gts-agarg35
#SBATCH --qos=embers
#SBATCH --partition=cpu-small
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.out
#SBATCH --error=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm/experiments/flow_2x2_20260604/slurm/%x-%j.err

set -euo pipefail

REPO=/storage/project/r-agarg35-0/eliu354/external_repos/le-wm
export STABLEWM_HOME=/storage/project/r-agarg35-0/eliu354/external_data/lewm_stablewm
export PYTHONNOUSERSITE=1

mkdir -p "${STABLEWM_HOME}/downloads" "${STABLEWM_HOME}/ogbench" "${STABLEWM_HOME}/experiments/flow_2x2_20260604/slurm"
cd "${REPO}"

PY="${REPO}/.conda/lewm-flow-2x2/bin/python"

"${PY}" scripts/download_cube_dataset.py

ARCHIVE="${STABLEWM_HOME}/downloads/cube_single_expert.tar.zst"
test -s "${ARCHIVE}"
tar --zstd -xvf "${ARCHIVE}" -C "${STABLEWM_HOME}"

if [ -f "${STABLEWM_HOME}/cube_single_expert.h5" ] && [ ! -f "${STABLEWM_HOME}/ogbench/cube_single_expert.h5" ]; then
  mv "${STABLEWM_HOME}/cube_single_expert.h5" "${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
fi

test -s "${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
echo "Cube dataset ready: ${STABLEWM_HOME}/ogbench/cube_single_expert.h5"
