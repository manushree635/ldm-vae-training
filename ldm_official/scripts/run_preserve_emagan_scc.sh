#!/bin/bash -l
# Export every EMA+GAN checkpoint as it appears, before save_top_k=3 prunes it.
#
# Necessary because the checkpoint monitor is val/rec_loss, which on this run is
# ANTI-correlated with quality: epoch 0 (no GAN) has the best reconstruction loss
# and the worst rFID. Left alone, Lightning would keep epoch 0 and discard the
# good ones. CPU-only job; runs alongside training.
#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=12:00:00
#$ -N emagan_keep
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -uo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
CK="${LDM_ROOT}/logs/2026-08-29T22-17-11_ldm_ftema_emagan_20k/checkpoints"
TRAIN_JOB="${TRAIN_JOB:-7367616}"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

while true; do
    for F in "${CK}"/epoch=*.ckpt; do
        [ -e "${F}" ] || continue
        EP=$(basename "${F}" .ckpt); EP=${EP#epoch=}
        for MODE in raw ema; do
            DEST="${LDM_ROOT}/models/emagan_ep${EP}_${MODE}"
            [ -d "${DEST}" ] && continue
            FLAG=""; [ "${MODE}" = "ema" ] && FLAG="--use-ema"
            echo "$(date +%H:%M:%S) exporting ep${EP} ${MODE}"
            "${PY}" scripts/export_ldm_vae_to_diffusers.py --ckpt "${F}" --out "${DEST}" ${FLAG} \
                >/dev/null 2>&1 || echo "  export failed (ckpt may have been pruned mid-read)"
        done
    done
    qstat -j "${TRAIN_JOB}" >/dev/null 2>&1 || { echo "training job ${TRAIN_JOB} ended"; break; }
    sleep 120
done
echo "preserved: $(ls -d ${LDM_ROOT}/models/emagan_ep*_raw 2>/dev/null | wc -l) epochs"
