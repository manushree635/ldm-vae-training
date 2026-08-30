#!/bin/bash -l
# Reconstruct through the EMA+GAN run against its own pre-GAN state, the plain
# run at matched epochs, and Stability's reference models.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=2:00:00
#$ -N recon_gan
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/recon_gan_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"

ARMS=(
  --arm "init-sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae"
  --arm "noGAN-1ep=${LDM_ROOT}/models/emagan_ep000000_raw"
  --arm "GAN-2ep=${LDM_ROOT}/models/emagan_ep000001_raw"
  --arm "GAN-2ep-ema=${LDM_ROOT}/models/emagan_ep000001_ema"
  --arm "plainLDM-2ep=${LDM_ROOT}/models/ldm_ftema_ep000001_diffusers"
  --arm "plainLDM-21ep=${LDM_ROOT}/models/ldm_ftema_ep000020_diffusers"
  --arm "stability-ft-ema=stabilityai/sd-vae-ft-ema"
  --arm "stability-ft-mse=stabilityai/sd-vae-ft-mse"
)
for IMG_NAME in ${IMAGES:-blue_camo gown}; do
  case "${IMG_NAME}" in
    blue_camo) IMG="${ROOT}/distillation/data/laion_faces_urls_downloaded_1000/Second-Phase14396.jpg" ;;
    gown)      IMG="${ROOT}/distillation/data/data_test_100/Second-Phase17674.jpg" ;;
    child)     IMG="${ROOT}/Experiments/200K_for_200Kids_blog_post_15nov2022.jpg" ;;
  esac
  echo "=== ${IMG_NAME} ==="
  "${PY}" scripts/recon_compare.py --image "${IMG}" --size 256 --squash \
      --out "${LDM_ROOT}/recon_compare/gan_${IMG_NAME}" "${ARMS[@]}"
done
