#!/bin/bash -l
# rFID for the 20k steerable+LPIPS distillation runs (teacher target, str 0.5).
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N rfid_strlpips
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"; LDM_ROOT="${ROOT}/ldm_official"
C=/projectnb/ivc-ml/mvasu6/checkpoints
B="${C}/teacher-512-student-256-laion-subset-20k-str-precomputed-0.5-lr-diffweight_lpips"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}" PYTHONUNBUFFERED=1
TRACE="${LDM_ROOT}/logs/rfid_strlpips_trace.log"; : > "${TRACE}"
exec > >(tee -a "${TRACE}") 2>&1
cd "${LDM_ROOT}"
"${ROOT}/venv_train/bin/python" scripts/eval_rfid_coco.py --control "sd-vae" \
  --arm "str-lpips-ep50=${B}/checkpoint_epoch_0050" \
  --arm "str-lpips-best=${B}/best_student_256" \
  --arm "str-lpips0.03-ep50=${B}-0.03/checkpoint_epoch_0050" \
  --arm "str-lpips0.03-best=${B}-0.03/best_student_256" \
  --out "${LDM_ROOT}/vae_bench_results" --tag strlpips
