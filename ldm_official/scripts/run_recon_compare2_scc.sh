#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=1:00:00
#$ -N recon_cmp2
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
NERC_CK="/projectnb/ivc-ml/mvasu6/checkpoints/mse-lpipsvgg-2gpu-20k"
STRPY="/projectnb/ivc-ml/mvasu6/checkpoints/teacher-512-student-256-laion-1M-precomputed-0.5-lr-diffweight_phase_fine_to_coarse/best_student_256"
IMG="${ROOT}/distillation/data/data_test_100/Second-Phase17674.jpg"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

ARMS=(
  --arm "sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae"
  --arm "sd-vae-ft-ema=stabilityai/sd-vae-ft-ema"
  --arm "sd-vae-ft-mse=stabilityai/sd-vae-ft-mse"
  --arm "strpy-1M=${STRPY}"
  --arm "nerc-vgg-ep10=${NERC_CK}/checkpoint_epoch_0010"
  --arm "ldm-ftema-ep4=${LDM_ROOT}/models/ldm_ftema_ep000004_diffusers"
)
# --squash == inference.py's Resize((S,S)). Two resolutions: 256 is the training
# and eval size; 512 is a genuine-detail pass (the VAE is fully convolutional),
# so the page can show real pixels instead of an upscale.
for S in 256 512; do
  echo "=== ${S}px ==="
  "${PY}" scripts/recon_compare.py --image "${IMG}" --size "${S}" --squash \
      --out "${LDM_ROOT}/recon_compare/red_dress_${S}" "${ARMS[@]}"
done
