#!/bin/bash -l
# rFID for the EMA+GAN run: raw weights vs EMA shadow, at each saved epoch.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_emagan
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
RUN="${LDM_ROOT}/logs/2026-08-29T22-17-11_ldm_ftema_emagan_20k"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/rfid_emagan_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"

ARMS=()
for CK in "${RUN}"/checkpoints/epoch=*.ckpt; do
    [ -e "${CK}" ] || continue
    EP=$(basename "${CK}" .ckpt); EP=${EP#epoch=}; N=$((10#${EP}+1))
    for MODE in raw ema; do
        DEST="${LDM_ROOT}/models/emagan_ep${EP}_${MODE}"
        FLAG=""; [ "${MODE}" = "ema" ] && FLAG="--use-ema"
        [ -d "${DEST}" ] || "${PY}" scripts/export_ldm_vae_to_diffusers.py \
            --ckpt "${CK}" --out "${DEST}" ${FLAG}
        ARMS+=(--arm "emagan-${N}ep-${MODE}=${DEST}")
    done
done
echo "arms: ${#ARMS[@]}"
"${PY}" scripts/eval_rfid_coco.py --control "sd-vae" "${ARMS[@]}" \
    --out "${LDM_ROOT}/vae_bench_results" --tag emagan
