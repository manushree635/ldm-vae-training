#!/bin/bash -l
# Train the dnarea arm: teacher @512 brought down to 256 by area averaging (= cv2.INTER_AREA)
# Both arms compare at 256, so the student's prediction is never resampled.
#
#   bash scripts_rebuttal/run_dnarea.sh                 # on a machine with 2 GPUs
#   SMOKE=1 bash scripts_rebuttal/run_dnarea.sh         # 12 steps first, ~5 min
#   qsub -P <project> -l gpus=2 -l gpu_c=8.0 scripts_rebuttal/run_dnarea.sh
#
# Thin wrapper over scripts/train.sh -- NUM_GPUS and EXTRA pass straight through, e.g.
#   NUM_GPUS=4 EXTRA="data.params.batch_size=24 lightning.trainer.accumulate_grad_batches=4"
#
# Needs the data in place first:
#   scripts_rebuttal/download_1M_data.sh  then  scripts_rebuttal/prepare_data_1M.sh
#$ -cwd
#$ -pe omp 16
#$ -l h_rt=24:00:00
#$ -N dnarea
#$ -j y
#$ -o logs/
set -euo pipefail
ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export LDM_ROOT="$ROOT"; cd "$ROOT"
CFG=configs/pretune_1M_dnarea.yaml
[ -f "$CFG" ] || { echo "missing $CFG -- are you in the repo root?" >&2; exit 1; }
if [ "${SMOKE:-0}" = "1" ]; then
    exec env CONFIG="$CFG" BATCH="${BATCH:-2}" bash scripts/smoke_test.sh
fi
exec env CONFIG="$CFG" NUM_GPUS="${NUM_GPUS:-2}" RUN_NAME="${RUN_NAME:-dnarea}" \
     EXTRA="${EXTRA:-}" bash scripts/train.sh
