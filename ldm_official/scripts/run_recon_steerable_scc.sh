#!/bin/bash -l
# Does the steerable pyramid term preserve high-frequency texture?
# rFID judged it on COCO with several confounds; texture retention is the metric
# that actually exposed the blur, so measure that directly.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=2:00:00
#$ -N recon_str
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"; C=/projectnb/ivc-ml/mvasu6/checkpoints
B="${C}/teacher-512-student-256-laion-subset-20k-str-precomputed-0.5-lr-diffweight_lpips"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/recon_str_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"
ARMS=(
  --arm "init-sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae"
  --arm "STR-lpips0.09=${B}/checkpoint_epoch_0050"
  --arm "STR-lpips0.03=${B}-0.03/checkpoint_epoch_0050"
  --arm "noSTR-mse0.1=${C}/mse-lpipsvgg-2gpu-20k/checkpoint_epoch_0036"
  --arm "noSTR-l1-1.0=${C}/l1_lpipsvgg1.0_sdvae_2gpu_20k/checkpoint_epoch_0050"
  --arm "GAN-2ep=${LDM_ROOT}/models/emagan_ep000001_raw"
  --arm "stability-ft-ema=stabilityai/sd-vae-ft-ema"
)
for N in blue_camo gown; do
  case "${N}" in
    blue_camo) IMG="${ROOT}/distillation/data/laion_faces_urls_downloaded_1000/Second-Phase14396.jpg";;
    gown)      IMG="${ROOT}/distillation/data/data_test_100/Second-Phase17674.jpg";;
  esac
  echo "=== ${N} ==="
  "${PY}" scripts/recon_compare.py --image "${IMG}" --size 256 --squash \
      --out "${LDM_ROOT}/recon_compare/str_${N}" "${ARMS[@]}"
done
