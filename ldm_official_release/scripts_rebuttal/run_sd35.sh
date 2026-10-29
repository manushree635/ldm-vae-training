#!/bin/bash -l
# Train the SD-3.5 arm: the nmn1M recipe with SD-3.5 Large's VAE as both student and teacher.
#
#   bash scripts_rebuttal/run_sd35.sh                  # on a machine with 2 GPUs
#   SMOKE=1 bash scripts_rebuttal/run_sd35.sh          # 12 steps first, ~5 min
#   qsub -P <project> -l gpus=2 -l gpu_c=8.0 scripts_rebuttal/run_sd35.sh
#
# Thin wrapper over scripts/train.sh -- NUM_GPUS and EXTRA pass straight through. If the
# live encoder forward (this arm has no latent cache) runs you out of memory, halve the
# micro-batch and double the accumulation so the global batch stays at 192:
#   EXTRA="data.params.batch_size=24 lightning.trainer.accumulate_grad_batches=4"
#
# Needs the data AND the SD-3.5 teacher set in place first:
#   scripts_rebuttal/download_1M_data.sh      then
#   scripts_rebuttal/prepare_sd35_teacher.sh  (NOT prepare_data_1M.sh -- different teacher,
#                                              different file lists, no latent cache)
#$ -cwd
#$ -pe omp 16
#$ -l h_rt=24:00:00
#$ -N sd35
#$ -j y
#$ -o logs/
set -euo pipefail
ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export LDM_ROOT="$ROOT"; cd "$ROOT"
CFG=configs/pretune_1M_sd35.yaml
[ -f "$CFG" ] || { echo "missing $CFG -- are you in the repo root?" >&2; exit 1; }
# Fail here with a sentence rather than inside the dataloader with a stack trace.
[ -f "$ROOT/models/sd35_stock_vae/config.json" ] || {
    echo "missing models/sd35_stock_vae -- run scripts_rebuttal/prepare_sd35_teacher.sh first" >&2; exit 1; }
[ -s "$ROOT/filelists/split_0_1M_sd35/train.txt" ] || {
    echo "missing filelists/split_0_1M_sd35 -- run scripts_rebuttal/prepare_sd35_teacher.sh first" >&2; exit 1; }
if [ "${SMOKE:-0}" = "1" ]; then
    exec env CONFIG="$CFG" BATCH="${BATCH:-2}" bash scripts/smoke_test.sh
fi
exec env CONFIG="$CFG" NUM_GPUS="${NUM_GPUS:-2}" RUN_NAME="${RUN_NAME:-sd35}" \
     EXTRA="${EXTRA:-}" bash scripts/train.sh
