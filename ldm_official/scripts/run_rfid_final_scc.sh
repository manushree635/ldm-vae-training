#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_final
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/rfid_final_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"
"${ROOT}/venv_train/bin/python" scripts/eval_rfid_coco.py --control "sd-vae" \
  --arm "ldm-19ep=${LDM_ROOT}/models/ldm_ftema_ep000018_diffusers" \
  --arm "ldm-21ep=${LDM_ROOT}/models/ldm_ftema_ep000020_diffusers" \
  --out "${LDM_ROOT}/vae_bench_results" --tag final
