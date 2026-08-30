#!/bin/bash -l
# Export LDM checkpoints to diffusers and score rFID on COCO with vae_bench.
#
#   qsub scripts/run_rfid_epoch_scc.sh
#
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N ldm_rfid
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs

set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
RUN="${RUN:-${LDM_ROOT}/logs/2026-08-29T13-39-04_ldm_ftema_20k}"
PY="${ROOT}/venv_train/bin/python"

# venv_train is torch cu128 -- no sm_70 kernels, hence gpu_c=7.5 above.
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}"
export PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"

echo "=== export checkpoints to diffusers ==="
"${PY}" scripts/export_ldm_vae_to_diffusers.py --self-test

ARMS=()
for CK in "${RUN}"/checkpoints/epoch=*.ckpt; do
    [ -e "${CK}" ] || continue
    EP="$(basename "${CK}" .ckpt)"; EP="${EP#epoch=}"
    DEST="${LDM_ROOT}/models/ldm_ftema_ep${EP}_diffusers"
    "${PY}" scripts/export_ldm_vae_to_diffusers.py --ckpt "${CK}" --out "${DEST}"
    ARMS+=(--arm "ldm-ftema-ep${EP}=${DEST}")
done

echo
echo "=== rFID on COCO val2017 (n=2000, 256px) ==="
"${PY}" scripts/eval_rfid_coco.py "${ARMS[@]}" \
    --out "${LDM_ROOT}/vae_bench_results" --tag ldm

echo
echo "=== results ==="
"${PY}" - <<'PY'
import json, glob
for f in sorted(glob.glob("/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/vae_bench_results/*recon*.json")):
    d = json.load(open(f))
    print(f)
    for k, v in d.get("rows", {}).items():
        print(f"  {k:<22} rFID={v['rFID']:.3f}  PSNR={v['PSNR']:.2f}  n={v['n']}")
PY
