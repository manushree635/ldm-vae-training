#!/bin/bash
# Activate the pinned LDM training environment (torch 1.11.0+cu113, PL 1.4.2).
#
# Sourcing this puts `ldm_ftmse` (the ft-MSE overlay) on PYTHONPATH and leaves
# the shell in latent-diffusion/, which is where main.py must run from: the YAML
# refers to `main.DataModuleFromConfig` and `main.ImageLogger`, so main.py has
# to be importable as a top-level module.
LDM_ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"

module load python3/3.8.16
module load cuda/11.3

# shellcheck disable=SC1091
source "${LDM_ROOT}/venv_ldm/bin/activate"
export PYTHONPATH="${LDM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LDM_ROOT
