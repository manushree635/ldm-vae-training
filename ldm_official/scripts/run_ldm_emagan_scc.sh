#!/bin/bash -l
# sd-vae-ft-ema recipe PLUS EMA weights and an early, PRETRAINED discriminator.
# Tests the two ingredients Stability had that our plain runs did not.
#
# Defaults to the ft-EMA variant (L1 + 1.0*LPIPS, upstream's loss verbatim).
# For ft-MSE (MSE + 0.1*LPIPS):
#   CONFIG=configs/autoencoder_kl_32x32x4_ftmse_20k.yaml RUN_NAME=ldm_ftmse_20k \
#       qsub scripts/run_ldm_ft_20k_scc.sh
#
#   qsub scripts/run_ldm_ft_20k_scc.sh
#
# Override anything at submit time, e.g. a 2-GPU run that keeps the 192 global
# batch by doubling accumulation:
#   NUM_GPUS=2 EXTRA="lightning.trainer.accumulate_grad_batches=8" \
#       qsub -l gpus=2 scripts/run_ldm_ft_20k_scc.sh
#
#$ -P ivc-ml
#$ -pe omp 16
#$ -l gpus=4
#$ -l gpu_c=8.0
#$ -l gpu_type=(A40|A6000|A100)
#$ -l h_rt=12:00:00
#$ -N ldm_emagan
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/logs

set -euo pipefail

LDM_ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"

# Requested as plain `gpu_c=8.0`, i.e. any GPU of compute capability >= 8.0.
# Be aware of what that pool contains: torch 1.11.0+cu113 ships SASS for
# sm_35..sm_86 and no PTX, so of the 84 hosts matching gpu_c>=8.0 only the 32
# A100 (sm_80) / A6000 / A40 (sm_86) ones can run it. L40, L40S and RTX6000ada
# are sm_89, H200 is sm_90, RTXP6000 is sm_120 -- those 52 hosts die on the
# first kernel launch. gpu_arch_guard.py below turns that into a clean abort in
# seconds; to avoid the landing entirely, add the allow-list, which SGE honours
# as an alternation in a RESTRING complex:
#   qsub -l 'gpu_type=(A40|A6000|A100)' scripts/run_ldm_ft_20k_scc.sh
: "${NUM_GPUS:=4}"
: "${CONFIG:=${LDM_ROOT}/configs/autoencoder_kl_32x32x4_ftema_emagan_20k.yaml}"
: "${RUN_NAME:=ldm_ftema_emagan_20k}"
: "${EXTRA:=}"

# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/activate_ldm_env.sh"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/scc_nccl_env.sh"
cd "${LDM_ROOT}/latent-diffusion"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-/projectnb/ivc-ml/mvasu6/.cache/huggingface}"
# LPIPS's VGG weights land in taming's package dir on first use; pre-warmed by
# scripts/smoke_test.sh so the four DDP ranks do not race to download them.
export TORCH_HOME="${TORCH_HOME:-/projectnb/ivc-ml/mvasu6/.cache/torch}"

GPUS="$(seq -s, 0 $((NUM_GPUS - 1))),"

echo "=== LDM ft-* decoder finetune ==="
echo "host        ${HOSTNAME:-unknown}"
echo "config      ${CONFIG}"
echo "gpus        ${GPUS} (${NUM_GPUS})"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python "${LDM_ROOT}/scripts/gpu_arch_guard.py"
python "${LDM_ROOT}/scripts/ddp_preflight.py" "${NUM_GPUS}"

# --scale_lr False: main.py otherwise sets lr = accum * ngpu * bs * base_lr,
# which would turn Stability's published 4.5e-6 into 8.6e-4. The model card's
# figure is the effective rate, so feed base_lr through unscaled.
exec python -u main.py \
    --base "${CONFIG}" \
    -t \
    --gpus "${GPUS}" \
    --scale_lr False \
    --name "${RUN_NAME}" \
    --logdir "${LDM_ROOT}/logs" \
    ${EXTRA}
