#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_nercfin
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
C=/projectnb/ivc-ml/mvasu6/checkpoints/l1_lpipsvgg1.0_sdvae_2gpu_20k
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/rfid_nercfinal_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"
ARMS=()
for EP in 0010 0020 0030 0040 0050; do
    [ -d "${C}/checkpoint_epoch_${EP}" ] && ARMS+=(--arm "nercL1-$((10#${EP}))ep=${C}/checkpoint_epoch_${EP}")
done
"${ROOT}/venv_train/bin/python" scripts/eval_rfid_coco.py --control "sd-vae" "${ARMS[@]}" \
    --out "${LDM_ROOT}/vae_bench_results" --tag nercfinal
