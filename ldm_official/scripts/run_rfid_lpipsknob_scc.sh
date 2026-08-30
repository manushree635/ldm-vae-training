#!/bin/bash -l
# rFID for the AlexNet-LPIPS lpipsknob family. These already vary the exact
# things we have been hypothesising about: data breadth (D), EMA (E, H),
# square crop vs squash (G), fp32 vs fp16 (C32 vs C), and LPIPS weight (A vs C).
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=6:00:00
#$ -N rfid_knob
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
C=/projectnb/ivc-ml/mvasu6/checkpoints
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/rfid_knob_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"

"${ROOT}/venv_train/bin/python" scripts/eval_rfid_coco.py --control "sd-vae,sd-vae-ft-mse" \
  --arm "knobA-w0.1-lr1e5=${C}/lpipsknob-A-alex-w0.1-lr1e-5-20ep/checkpoint_epoch_0020" \
  --arm "knobB-w0.1-lr3e6=${C}/lpipsknob-B-alex-w0.1-lr3e-6-20ep/checkpoint_epoch_0020" \
  --arm "knobC-w0.004=${C}/lpipsknob-C-alex-w0.004-lr1e-5-20ep/checkpoint_epoch_0020" \
  --arm "knobC32-w0.004-fp32=${C}/lpipsknob-C32-alex-w0.004-lr1e-5-20ep/checkpoint_epoch_0020" \
  --arm "knobD-broad100k-4ep=${C}/lpipsknob-D-broaddata-100k-4ep/checkpoint_epoch_0004" \
  --arm "knobE-ema999=${C}/lpipsknob-E-ema999-20ep/checkpoint_epoch_0020" \
  --arm "knobE-ema999-EMA=${C}/lpipsknob-E-ema999-20ep/checkpoint_epoch_0020_ema" \
  --arm "knobF-freezedec=${C}/lpipsknob-F-freezedec-20ep/checkpoint_epoch_0020" \
  --arm "knobG-squarecrop=${C}/lpipsknob-G-squarecrop-20ep/checkpoint_epoch_0020" \
  --arm "knobH-freeze-ema=${C}/lpipsknob-H-freeze-lr3e-6-ema-20ep/checkpoint_epoch_0020" \
  --arm "knobH-freeze-ema-EMA=${C}/lpipsknob-H-freeze-lr3e-6-ema-20ep/checkpoint_epoch_0020_ema" \
  --out "${LDM_ROOT}/vae_bench_results" --tag knob
