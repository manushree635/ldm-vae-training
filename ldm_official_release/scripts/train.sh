#!/bin/bash -l
# Train any config. Runs the same way interactively and under SGE.
#
#   # on a machine with GPUs
#   CONFIG=configs/04_steerable_graded.yaml NUM_GPUS=2 bash scripts/train.sh
#
#   # BU SCC -- submit FROM THE REPO ROOT (-cwd is how the job finds the repo)
#   qsub -P <project> -N w1graded -l gpus=2 -l gpu_c=8.0 \
#        -v CONFIG=configs/04_steerable_graded.yaml,NUM_GPUS=2 scripts/train.sh
#
# NB on SGE the variables MUST go through `-v`. `CONFIG=x qsub scripts/train.sh`
# does not forward them, and the job silently runs the default config instead.
#
#   CONFIG     yaml to train (default configs/01_ftema_baseline.yaml); relative paths
#              resolve against the repo root
#   RUN_NAME   suffix of the run dir, logs/<timestamp>_<RUN_NAME> (default: the config's basename)
#   NUM_GPUS   DDP world size (default 1). Keep `-l gpus=` equal to it.
#   EXTRA      extra main.py args / OmegaConf dotlist overrides, space separated, e.g.
#              EXTRA="data.params.batch_size=8 lightning.trainer.max_epochs=10"
#   RESUME_DIR resume a killed run from its logs/<run> directory instead of starting one
#
#$ -cwd
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=24:00:00
#$ -N ldm_vae_train
#$ -j y
#$ -o logs/
set -euo pipefail

export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/nccl_env.sh"

: "${CONFIG:=configs/01_ftema_baseline.yaml}"
: "${NUM_GPUS:=1}"
: "${EXTRA:=}"
: "${RESUME_DIR:=}"
[[ "${CONFIG}" = /* ]] || CONFIG="${LDM_ROOT}/${CONFIG}"
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }
: "${RUN_NAME:=$(basename "${CONFIG}" .yaml)}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# DDP rendezvous port derived from the job id, so two jobs landing on the same
# node cannot collide on Lightning's fixed default. Must be set BEFORE the
# preflight, which reads it too.
export MASTER_PORT="${MASTER_PORT:-$(( 20000 + ${JOB_ID:-$$} % 20000 ))}"
GPUS="$(seq -s, 0 $((NUM_GPUS - 1))),"

echo "=== LDM VAE finetune ==="
echo "host    ${HOSTNAME:-unknown}"
echo "root    ${LDM_ROOT}"
echo "data    ${LDM_DATA}"
echo "config  ${CONFIG}"
echo "run     ${RUN_NAME}${RESUME_DIR:+   (resuming ${RESUME_DIR})}"
echo "gpus    ${GPUS} (${NUM_GPUS})"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
python "${LDM_ROOT}/scripts/check_config_paths.py" "${CONFIG}"
python "${LDM_ROOT}/scripts/gpu_arch_guard.py"
python "${LDM_ROOT}/scripts/ddp_preflight.py" "${NUM_GPUS}"

# main.py must run from latent-diffusion/: the YAML refers to main.DataModuleFromConfig
# and main.ImageLogger, and taming resolves its LPIPS weights relative to the cwd.
cd "${LDM_ROOT}/latent-diffusion"

# --scale_lr False: main.py otherwise sets lr = accum * ngpu * bs * base_lr, which
# turns 4.5e-6 into ~1e-3 -- right for training kl-f8 from scratch, destructive
# for a decoder finetune. base_learning_rate in the YAML is the rate actually used.
if [ -n "${RESUME_DIR}" ]; then
    [[ "${RESUME_DIR}" = /* ]] || RESUME_DIR="${LDM_ROOT}/${RESUME_DIR}"
    # --resume takes the RUN DIRECTORY and forbids --name; it reuses the logdir and
    # its saved configs, so epoch numbering continues.
    # shellcheck disable=SC2086
    exec python -u main.py --resume "${RESUME_DIR}" -t --gpus "${GPUS}" --scale_lr False \
        --logdir "${LDM_ROOT}/logs" ${EXTRA}
fi
# shellcheck disable=SC2086
exec python -u main.py --base "${CONFIG}" -t --gpus "${GPUS}" --scale_lr False \
    --name "${RUN_NAME}" --logdir "${LDM_ROOT}/logs" ${EXTRA}
