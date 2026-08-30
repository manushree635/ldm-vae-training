#!/bin/bash -l
# rFID trajectory for the LDM ft-EMA run: fill in epochs 7 and 12.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_traj
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
RUN="${LDM_ROOT}/logs/2026-08-29T13-39-04_ldm_ftema_20k"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

ARMS=()
for EP in 000007 000012; do
    CK="${RUN}/checkpoints/epoch=${EP}.ckpt"
    [ -e "${CK}" ] || { echo "skip: ${CK} gone (pruned by save_top_k)"; continue; }
    DEST="${LDM_ROOT}/models/ldm_ftema_ep${EP}_diffusers"
    "${PY}" scripts/export_ldm_vae_to_diffusers.py --ckpt "${CK}" --out "${DEST}"
    ARMS+=(--arm "ldm-ftema-ep${EP}=${DEST}")
done

"${PY}" scripts/eval_rfid_coco.py --control "sd-vae" "${ARMS[@]}" \
    --out "${LDM_ROOT}/vae_bench_results" --tag traj
