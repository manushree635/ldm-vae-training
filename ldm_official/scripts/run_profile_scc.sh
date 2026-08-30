#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=(A40|A6000|A100)
#$ -l h_rt=1:00:00
#$ -N ldm_profile
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
LDM_ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"
source "${LDM_ROOT}/scripts/activate_ldm_env.sh"
export PYTHONUNBUFFERED=1 TORCH_HOME="${TORCH_HOME:-/projectnb/ivc-ml/mvasu6/.cache/torch}"
cd "${LDM_ROOT}/latent-diffusion"
python "${LDM_ROOT}/scripts/profile_step.py"
