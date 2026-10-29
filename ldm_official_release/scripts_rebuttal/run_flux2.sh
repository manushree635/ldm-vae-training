#!/bin/bash -l
# Train the FLUX.2 arm: the nmn1M recipe with FLUX.2's VAE as both student and teacher.
#
#   bash scripts_rebuttal/run_flux2.sh                 # on a machine with 2 GPUs
#   SMOKE=1 bash scripts_rebuttal/run_flux2.sh         # 12 steps first, ~5 min
#   qsub -P <project> -l gpus=2 -l gpu_c=8.0 scripts_rebuttal/run_flux2.sh
#
# Thin wrapper over scripts/train.sh -- NUM_GPUS and EXTRA pass straight through. This arm
# pays both a live encoder forward (no latent cache) and a 32-channel latent, so it is the
# likeliest to OOM; halve the micro-batch and double accumulation to hold the global batch:
#   EXTRA="data.params.batch_size=24 lightning.trainer.accumulate_grad_batches=4"
#
# Sets EXTRA_PYTHONPATH itself: AutoencoderKLFlux2 does not exist in the pinned diffusers
# 0.30.3. Override it if your overlay lives elsewhere, or point it at a venv that already
# has diffusers >= 0.36.
#
# Needs, in order:
#   scripts_rebuttal/install_diffusers036.sh      once
#   scripts_rebuttal/download_1M_data.sh          the GT images
#   scripts_rebuttal/prepare_flux2_teacher.sh     NOT prepare_data_1M.sh -- different
#                                                 teacher, different lists, no latent cache
#$ -cwd
#$ -pe omp 16
#$ -l h_rt=24:00:00
#$ -N flux2
#$ -j y
#$ -o logs/
set -euo pipefail
ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export LDM_ROOT="$ROOT"; cd "$ROOT"
export EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-$ROOT/pylibs/diffusers036}"
CFG=configs/pretune_1M_flux2.yaml
[ -f "$CFG" ] || { echo "missing $CFG -- are you in the repo root?" >&2; exit 1; }
# Fail here with a sentence rather than inside the dataloader with a stack trace.
[ -d "$EXTRA_PYTHONPATH/diffusers" ] || {
    echo "no diffusers overlay at $EXTRA_PYTHONPATH -- run scripts_rebuttal/install_diffusers036.sh" >&2; exit 1; }
[ -f "$ROOT/models/flux2_stock_vae/config.json" ] || {
    echo "missing models/flux2_stock_vae -- run scripts_rebuttal/prepare_flux2_teacher.sh first" >&2; exit 1; }
[ -s "$ROOT/filelists/split_0_1M_flux2/train.txt" ] || {
    echo "missing filelists/split_0_1M_flux2 -- run scripts_rebuttal/prepare_flux2_teacher.sh first" >&2; exit 1; }
if [ "${SMOKE:-0}" = "1" ]; then
    exec env CONFIG="$CFG" BATCH="${BATCH:-2}" bash scripts/smoke_test.sh
fi
exec env CONFIG="$CFG" NUM_GPUS="${NUM_GPUS:-2}" RUN_NAME="${RUN_NAME:-flux2}" \
     EXTRA="${EXTRA:-}" bash scripts/train.sh
