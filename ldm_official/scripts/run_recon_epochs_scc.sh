#!/bin/bash -l
# Reconstruct one image through every preserved LDM checkpoint, to see how the
# decoder drifts with training.
#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=2:00:00
#$ -N recon_epochs
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
PY="${ROOT}/venv_train/bin/python"
IMG="${IMG:-${ROOT}/distillation/data/laion_faces_urls_downloaded_1000/Second-Phase14396.jpg}"
OUT="${OUT:-${LDM_ROOT}/recon_compare/blue_camo_256}"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

# SGE only flushes its -o file at job end, so a kill loses everything. Tee to a
# file we control, written as we go.
TRACE="${LDM_ROOT}/logs/recon_epochs_trace.log"
exec > >(tee -a "${TRACE}") 2>&1
echo "=== start $(date) on ${HOSTNAME:-?} ==="

# Convert anything preserved as a raw .ckpt that has no diffusers copy yet.
for CK in models/_preserved/epoch=*.ckpt; do
    [ -e "${CK}" ] || continue
    EP=$(basename "${CK}" .ckpt); EP=${EP#epoch=}
    DEST="models/ldm_ftema_ep${EP}_diffusers"
    [ -d "${DEST}" ] || "${PY}" scripts/export_ldm_vae_to_diffusers.py --ckpt "${CK}" --out "${DEST}"
done

# Arm 01 is the run's own init, so epoch drift reads against a fixed anchor.
ARMS=(--arm "init-sd-vae=stable-diffusion-v1-5/stable-diffusion-v1-5,vae")
for D in $(ls -d models/ldm_ftema_ep*_diffusers | sort); do
    EP=$(echo "${D}" | sed 's/.*_ep0*\([0-9]\+\)_diffusers/\1/')
    ARMS+=(--arm "ldm-ep${EP}=${LDM_ROOT}/${D}")
done
echo "arms: ${ARMS[*]}"
"${PY}" scripts/recon_compare.py --image "${IMG}" --size 256 --squash --out "${OUT}" "${ARMS[@]}"
