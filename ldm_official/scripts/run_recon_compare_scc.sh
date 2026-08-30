#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=1:00:00
#$ -N recon_cmp
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
RUN="${LDM_ROOT}/logs/2026-08-29T13-39-04_ldm_ftema_20k"
NERC_CK="/projectnb/ivc-ml/mvasu6/checkpoints/mse-lpipsvgg-2gpu-20k"
IMG="${ROOT}/distillation/data/data_test_100/Second-Phase17674.jpg"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

# Export the LDM epoch-4 checkpoint now: save_top_k=3 will eventually delete it.
"${PY}" scripts/export_ldm_vae_to_diffusers.py \
    --ckpt "${RUN}/checkpoints/epoch=000004.ckpt" \
    --out  "${LDM_ROOT}/models/ldm_ftema_ep000004_diffusers"

"${PY}" scripts/recon_compare.py --image "${IMG}" --size 256 \
    --out "${LDM_ROOT}/recon_compare/red_dress" \
    --arm "sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae" \
    --arm "sd-vae-ft-ema=stabilityai/sd-vae-ft-ema" \
    --arm "sd-vae-ft-mse=stabilityai/sd-vae-ft-mse" \
    --arm "strpy-1M=/projectnb/ivc-ml/mvasu6/checkpoints/teacher-512-student-256-laion-1M-precomputed-0.5-lr-diffweight_phase_fine_to_coarse/best_student_256" \
    --arm "nerc-vgg-ep10=${NERC_CK}/checkpoint_epoch_0010" \
    --arm "ldm-ftema-ep4=${LDM_ROOT}/models/ldm_ftema_ep000004_diffusers"
