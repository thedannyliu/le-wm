#!/bin/bash
set -euo pipefail

: "${STABLEWM_HOME:?STABLEWM_HOME must be set before sourcing slurm_runtime_env.sh}"

export LEWM_RUNTIME_ROOT=${LEWM_RUNTIME_ROOT:-${STABLEWM_HOME}/runtime}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${LEWM_RUNTIME_ROOT}/xdg_cache}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-${LEWM_RUNTIME_ROOT}/xdg_config}
export HF_HOME=${HF_HOME:-${LEWM_RUNTIME_ROOT}/hf_home}
export TORCH_HOME=${TORCH_HOME:-${LEWM_RUNTIME_ROOT}/torch_home}
export MPLCONFIGDIR=${MPLCONFIGDIR:-${LEWM_RUNTIME_ROOT}/matplotlib}
export WANDB_DIR=${WANDB_DIR:-${LEWM_RUNTIME_ROOT}/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${LEWM_RUNTIME_ROOT}/wandb_cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-${LEWM_RUNTIME_ROOT}/wandb_config}
export TMPDIR=${TMPDIR:-${LEWM_RUNTIME_ROOT}/tmp/${SLURM_JOB_ID:-manual}}
export LEWM_JOB_WORKDIR=${LEWM_JOB_WORKDIR:-${LEWM_RUNTIME_ROOT}/workdirs/${SLURM_JOB_ID:-manual}}

mkdir -p \
  "${XDG_CACHE_HOME}" \
  "${XDG_CONFIG_HOME}" \
  "${HF_HOME}" \
  "${TORCH_HOME}" \
  "${MPLCONFIGDIR}" \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${TMPDIR}" \
  "${LEWM_JOB_WORKDIR}"
