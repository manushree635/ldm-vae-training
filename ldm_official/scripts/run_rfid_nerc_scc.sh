#!/bin/bash -l
# rFID on COCO for distillation_nerc checkpoints, via the same vae_bench harness.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=4:00:00
#$ -N nerc_rfid
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs
set -euo pipefail
ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE"
LDM_ROOT="${ROOT}/ldm_official"
D="/projectnb/ivc-ml/mvasu6/checkpoints/mse-lpipsvgg-2gpu-20k"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}"
export PYTHONUNBUFFERED=1
cd "${LDM_ROOT}"
"${ROOT}/venv_train/bin/python" scripts/eval_rfid_coco.py \
    --control "sd-vae,sd-vae-ft-mse" \
    --arm "nerc-vgg-last-ep36=${D}/checkpoint_epoch_0036" \
    --arm "nerc-vgg-best=${D}/best_student_256" \
    --out "${LDM_ROOT}/vae_bench_results" --tag nerc
