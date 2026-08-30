#!/bin/bash -l
# End-to-end check before committing a GPU-day to the real run.
#
#   qsub scripts/smoke_test.sh
#
# 1. proves the diffusers -> LDM checkpoint conversion is numerically exact
# 2. runs a handful of real training steps on one GPU, which also pre-warms the
#    LPIPS VGG download so the 4-way DDP run does not race on it
#
#$ -P ivc-ml
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_type=A40
#$ -l h_rt=1:00:00
#$ -N ldm_ftmse_smoke
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs

set -euo pipefail

LDM_ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"
CONFIG="${CONFIG:-${LDM_ROOT}/configs/autoencoder_kl_32x32x4_ftema_20k.yaml}"
# BATCH=12 turns this into a memory probe for the real run's per-GPU batch;
# watch the "Average Peak memory" line the CUDACallback prints at epoch end.
: "${BATCH:=4}"

# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/activate_ldm_env.sh"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/projectnb/ivc-ml/mvasu6/.cache/torch}"

echo "############ 1/2  checkpoint conversion ############"
cd "${LDM_ROOT}"
python scripts/verify_converted_vae.py \
    --ckpt "${LDM_ROOT}/models/sd15_vae_ldm.ckpt" \
    --config "${CONFIG}"

echo
echo "############ 2/2  training steps ############"
cd "${LDM_ROOT}/latent-diffusion"
python -u main.py \
    --base "${CONFIG}" \
    -t \
    --gpus "0," \
    --scale_lr False \
    --name ldm_ftmse_smoke \
    --logdir "${LDM_ROOT}/logs" \
    --max_steps 12 \
    --limit_val_batches 2 \
    lightning.trainer.accumulate_grad_batches=1 \
    lightning.callbacks.image_logger.params.batch_frequency=4 \
    data.params.batch_size="${BATCH}" \
    data.params.num_workers=2

echo
echo "smoke test finished"
