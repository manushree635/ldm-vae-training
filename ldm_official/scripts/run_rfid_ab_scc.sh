#!/bin/bash -l
# rFID on COCO for the framework A/B: same recipe (L1 + 1.0*LPIPS, init sd-vae),
# distillation_nerc trainer vs CompVis/latent-diffusion.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_ab
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
RUN="${LDM_ROOT}/logs/2026-08-29T13-39-04_ldm_ftema_20k"
NERC="/projectnb/ivc-ml/mvasu6/checkpoints/l1_lpipsvgg1.0_sdvae_2gpu_20k"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

# Export the LDM epoch-9 checkpoint before save_top_k=3 prunes it.
"${PY}" scripts/export_ldm_vae_to_diffusers.py \
    --ckpt "${RUN}/checkpoints/epoch=000009.ckpt" \
    --out  "${LDM_ROOT}/models/ldm_ftema_ep000009_diffusers"

"${PY}" scripts/eval_rfid_coco.py \
    --control "sd-vae,sd-vae-ft-mse" \
    --arm "nerc-l1-ep2=${NERC}/checkpoint_epoch_0002" \
    --arm "ldm-ftema-ep9=${LDM_ROOT}/models/ldm_ftema_ep000009_diffusers" \
    --out "${LDM_ROOT}/vae_bench_results" --tag ab
