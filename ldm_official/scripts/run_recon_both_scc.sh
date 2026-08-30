#!/bin/bash -l
# Reconstruct one image through BOTH runs' checkpoints, and rFID the nerc arm.
# Labels are epochs-COMPLETED: LDM's epoch=N.ckpt is after N+1 epochs (PL is
# 0-indexed); nerc's checkpoint_epoch_N is after N epochs (1-indexed).
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N recon_both
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
NERC="/projectnb/ivc-ml/mvasu6/checkpoints/l1_lpipsvgg1.0_sdvae_2gpu_20k"
IMG="${IMG:-${ROOT}/distillation/data/laion_faces_urls_downloaded_1000/Second-Phase14396.jpg}"
OUT="${OUT:-${LDM_ROOT}/recon_compare/blue_camo_both}"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${TRACE:-${LDM_ROOT}/logs/recon_both_trace.log}"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"
echo "=== start $(date) on ${HOSTNAME:-?} ==="

ARMS=(--arm "init-sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae")
for D in $(ls -d models/ldm_ftema_ep*_diffusers | sort); do
    N=$(echo "${D}" | sed 's/.*_ep0*\([0-9]\+\)_diffusers/\1/'); N=${N:-0}
    ARMS+=(--arm "ldm-$((N+1))ep=${LDM_ROOT}/${D}")
done
NERC_ARMS=()
for C in $(ls -d ${NERC}/checkpoint_epoch_* 2>/dev/null | sort -V); do
    E=$(basename "${C}" | sed 's/checkpoint_epoch_0*//')
    ARMS+=(--arm "nerc-${E}ep=${C}")
    NERC_ARMS+=(--arm "nerc-${E}ep=${C}")
done

echo "--- reconstructions ---"
"${PY}" scripts/recon_compare.py --image "${IMG}" --size 256 --squash --out "${OUT}" "${ARMS[@]}"

if [ "${SKIP_RFID:-0}" != "1" ]; then
  echo "--- rFID for the nerc arm ---"
  "${PY}" scripts/eval_rfid_coco.py --control "sd-vae" "${NERC_ARMS[@]}" \
      --out "${LDM_ROOT}/vae_bench_results" --tag nerctraj
fi
