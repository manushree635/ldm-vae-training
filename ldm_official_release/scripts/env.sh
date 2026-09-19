#!/bin/bash
# Source this (do not execute it) to get a shell that can run everything here:
#
#   source scripts/env.sh
#
# It works from any clone location: LDM_ROOT is derived from this file's own
# path, and every config reaches its checkpoints, file lists and data through
# ${oc.env:LDM_ROOT} / ${oc.env:LDM_DATA} rather than an absolute path.
#
#   LDM_ROOT   this directory (set for you)
#   LDM_DATA   where the image sets and teacher reconstructions live
#              (default: $LDM_ROOT/data -- export it BEFORE sourcing to override)
#   LDM_VENV   the virtualenv to activate (default: $LDM_ROOT/venv)
#
# Under `qsub` the job script is spooled to a temp dir, so BASH_SOURCE does not
# point into the repo there. The launchers pass LDM_ROOT in explicitly, which
# is why it is only derived here when it is not already set.
if [ -z "${LDM_ROOT:-}" ]; then
    LDM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
if [ ! -f "${LDM_ROOT}/latent-diffusion/main.py" ]; then
    echo "env.sh: LDM_ROOT=${LDM_ROOT} does not look like this repo (no latent-diffusion/main.py)" >&2
    return 1 2>/dev/null || exit 1
fi
export LDM_ROOT
export LDM_DATA="${LDM_DATA:-${LDM_ROOT}/data}"
: "${LDM_VENV:=${LDM_ROOT}/venv}"

# BU SCC (or any Environment-Modules cluster): load the interpreter and CUDA the
# venv was built against. Harmless no-op everywhere else. Override the module
# names with LDM_PY_MODULE / LDM_CUDA_MODULE if your site spells them differently.
# setup_env.sh records the pair it used in venv/ldm_modules.sh (cu121 vs cu128 differ).
# shellcheck disable=SC1091
[ -f "${LDM_VENV}/ldm_modules.sh" ] && source "${LDM_VENV}/ldm_modules.sh"
if command -v module >/dev/null 2>&1; then
    module load "${LDM_PY_MODULE:-python3/3.9.9}" 2>/dev/null || true
    module load "${LDM_CUDA_MODULE:-cuda/12.2}" 2>/dev/null || true
fi

if [ ! -f "${LDM_VENV}/bin/activate" ]; then
    echo "env.sh: no virtualenv at ${LDM_VENV} -- run  bash setup_env.sh  first" >&2
    return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "${LDM_VENV}/bin/activate"

# ldm_ftmse (the overlay) is imported as a top-level package by the YAML
# `target:` strings and by latent-diffusion/main.py; ldm and taming have no
# __init__.py (implicit namespace packages), so they go on the path wholesale.
export PYTHONPATH="${LDM_ROOT}:${LDM_ROOT}/latent-diffusion:${LDM_ROOT}/taming-transformers${PYTHONPATH:+:${PYTHONPATH}}"

# Weight caches. Left alone if you already export them.
export HF_HOME="${HF_HOME:-${LDM_ROOT}/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${LDM_ROOT}/.cache/torch}"
export PYTHONUNBUFFERED=1
