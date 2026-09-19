#!/bin/bash -l
# 12 real training steps on one GPU -- run this before committing a GPU-day.
#
#   CONFIG=configs/04_steerable_graded.yaml bash scripts/smoke_test.sh
#   qsub -P <project> -v CONFIG=configs/04_steerable_graded.yaml scripts/smoke_test.sh
#
# It exercises the whole path (env, config paths, GPU arch, data, model init from
# the checkpoint, both optimizers, validation, image logging, checkpoint write) and
# pre-warms the LPIPS VGG download so multi-GPU ranks do not race on it later.
# BATCH=<per-GPU batch of the real run> turns it into a memory probe: read the
# "Average Peak memory" line printed at the end of the epoch.
#$ -cwd
#$ -pe omp 4
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=1:00:00
#$ -N ldm_vae_smoke
#$ -j y
#$ -o logs/
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
: "${CONFIG:=configs/01_ftema_baseline.yaml}"
: "${BATCH:=4}"
[[ "${CONFIG}" = /* ]] || CONFIG="${LDM_ROOT}/${CONFIG}"
# Never let a smoke test create a wandb run.
export WANDB_MODE="${WANDB_MODE:-disabled}"

python "${LDM_ROOT}/scripts/check_config_paths.py" "${CONFIG}"
python "${LDM_ROOT}/scripts/gpu_arch_guard.py"
cd "${LDM_ROOT}/latent-diffusion"
python -u main.py --base "${CONFIG}" -t --gpus "0," --scale_lr False \
    --name "smoke_$(basename "${CONFIG}" .yaml)" --logdir "${LDM_ROOT}/logs" \
    --max_steps 12 --log_every_n_steps 1 --limit_val_batches 2 \
    lightning.trainer.accumulate_grad_batches=1 \
    data.params.batch_size="${BATCH}" \
    data.params.num_workers=2
echo
echo "smoke test finished OK -- delete logs/*_smoke_* when you are done looking at it"
