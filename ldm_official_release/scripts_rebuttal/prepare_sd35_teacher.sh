#!/bin/bash -l
# Generate the SD-3.5 TEACHER output for the 1M set, then the file lists that go with it.
# This is prepare_data_1M.sh's teacher pass with the model swapped: same images, same
# Resize((512,512)) squash, same posterior-mean round trip, SD-3.5 Large's VAE instead of
# sd-vae-ft-mse. Feeds configs/pretune_1M_sd35.yaml.
#
#   bash scripts_rebuttal/prepare_sd35_teacher.sh
#   qsub -P <project> -l gpus=4 -l gpu_c=8.0 -pe omp 16 scripts_rebuttal/prepare_sd35_teacher.sh
#
#   SRC_DIR=...   image folder, if $LDM_DATA/split_0 is not already there
#   NUM_SHARDS=   processes, one per granted GPU (default: as many GPUs as are visible)
#   BATCH=8  WORKERS=8  VAL_N=2000
#   VAE_REPO=     where the VAE lives (default $LDM_ROOT/models/sd35_stock_vae, fetched below)
#
# Produces:
#   $LDM_ROOT/models/sd35_stock_vae                     the VAE itself, ~320 MB (once)
#   $LDM_DATA/teacher_recons_512_png_sd35               one 512 PNG per image, ~350 GB
#   $LDM_ROOT/filelists/split_0_1M_sd35/{train,val}.txt teacher-intersected, val_n 2000
#
# WHY A SEPARATE TEACHER SET. The steerable term pulls the student towards what the teacher
# reconstructs at 512. For the SD-3.5 arm the teacher IS SD-3.5, so ft-mse's recons are the
# wrong target -- they would cap the term at a weaker VAE's output. And a separate pass drops
# a different handful of undecodable images, so the file lists must be rebuilt too, not
# reused from split_0_1M.
#
# WHY NO LATENT CACHE STEP. prepare_data_1M.sh ends by caching the frozen encoder's output.
# That cache is sd-vae-ft-mse's 4-channel latents; SD-3.5's encoder is a different function
# with a 16-channel output, and the model class for this arm has no cache path at all. The
# arm pays one encoder forward per step instead. Nothing to build here.
#
# COST AND RESUMING. The teacher pass dominates: SD-3.5's VAE is the same f8 size as kl-f8,
# so budget roughly the ~1 GPU-day prepare_data_1M.sh quotes for 1M images, divided by
# NUM_SHARDS. That will exceed a 12 h wall clock on few GPUs, which is fine -- the script
# skips images whose PNG already exists, so resubmit until the count stops moving. Shards are
# re-split across the REMAINING work on each resubmission, so they stay balanced.
#
# DISK. ~350 GB for the recons, on top of whatever the GT and the ft-mse teacher set already
# take. Check before starting; a full filesystem mid-pass leaves truncated PNGs, which
# decode as errors at train time rather than being skipped.
#$ -cwd
#$ -pe omp 16
#$ -l h_rt=24:00:00
#$ -N sd35_teacher
#$ -j y
#$ -o logs/
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
cd "${LDM_ROOT}"

: "${VAE_REPO:=${LDM_ROOT}/models/sd35_stock_vae}"
: "${BATCH:=8}"
: "${WORKERS:=8}"
: "${VAL_N:=2000}"
OUT="${LDM_DATA}/teacher_recons_512_png_sd35"
LISTS=filelists/split_0_1M_sd35
mkdir -p "${LDM_DATA}" "${OUT}" "${LISTS}" logs

if [ ! -e "${LDM_DATA}/split_0" ]; then
    : "${SRC_DIR:?set SRC_DIR to your image folder (or put the images at ${LDM_DATA}/split_0)}"
    ln -s "$(cd "${SRC_DIR}" && pwd)" "${LDM_DATA}/split_0"
fi
N_IMG=$(ls "${LDM_DATA}/split_0" | wc -l)
echo "images: ${N_IMG} under ${LDM_DATA}/split_0"

# One process per GPU the scheduler actually granted. Hardcoding 0..N-1 would run on GPUs
# this job does not own; CUDA_VISIBLE_DEVICES is the list we are allowed to touch.
GRANTED=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GRANTED <<< "${CUDA_VISIBLE_DEVICES}"
else
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    [ "${n}" -lt 1 ] && n=1
    for ((g = 0; g < n; g++)); do GRANTED+=("${g}"); done
fi
: "${NUM_SHARDS:=${#GRANTED[@]}}"
if [ "${NUM_SHARDS}" -gt "${#GRANTED[@]}" ]; then
    echo "FATAL: NUM_SHARDS=${NUM_SHARDS} but only ${#GRANTED[@]} GPU(s) available (${CUDA_VISIBLE_DEVICES:-none})" >&2
    exit 1
fi

echo "### 1/3 the SD-3.5 VAE itself"
if [ -f "${VAE_REPO}/config.json" ]; then
    echo "have it: ${VAE_REPO}"
else
    # stabilityai/stable-diffusion-3.5-large is gated; this mirror of its vae/ is not, and
    # the weights are identical. Set VAE_REPO to a local copy to skip the download.
    echo "fetching into ${VAE_REPO} (~320 MB)"
    mkdir -p "${VAE_REPO}"
    python - "$VAE_REPO" <<'PY'
import sys
from diffusers import AutoencoderKL
dst = sys.argv[1]
vae = AutoencoderKL.from_pretrained("adamo1139/stable-diffusion-3.5-large-ungated", subfolder="vae")
vae.save_pretrained(dst)
print("latent_channels =", vae.config.latent_channels, "->", dst)
PY
fi

echo "### 2/3 teacher recons at 512 (the long one), ${NUM_SHARDS} shard(s) -> ${OUT}"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    dev="${GRANTED[$i]}"
    echo "--- shard ${i}/${NUM_SHARDS} on GPU ${dev}"
    CUDA_VISIBLE_DEVICES="${dev}" python scripts/precompute_teacher_recons.py \
        --teacher "${VAE_REPO}" \
        --src_dir "${LDM_DATA}/split_0" \
        --out_dir "${OUT}" \
        --size 512 --batch_size "${BATCH}" --num_workers "${WORKERS}" \
        --num_shards "${NUM_SHARDS}" --shard_index "${i}" \
        > "logs/sd35_teacher_shard${i}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
tail -n 2 logs/sd35_teacher_shard*.log || true
HAVE=$(find "${OUT}" -maxdepth 1 -name '*.png' | wc -l)
echo "recons present: ${HAVE} / ${N_IMG}"
if [ "${status}" -ne 0 ]; then echo "FATAL: a shard failed -- see logs/sd35_teacher_shard*.log" >&2; exit 1; fi
if [ "${HAVE}" -lt "${N_IMG}" ]; then
    # Expected on the first submission: this is more than a wall clock's work.
    echo "INCOMPLETE: resubmit this script to continue (finished images are not redone)."
    echo "Stopping before the file lists, which must describe a COMPLETE teacher set."
    exit 0
fi

echo "### 3/3 file lists, keeping only images that have an SD-3.5 recon"
python scripts/make_filelists_teacher.py \
    --data_dir    "${LDM_DATA}/split_0" \
    --teacher_dir "${OUT}" \
    --out_dir     "${LISTS}" --val_n "${VAL_N}"
wc -l "${LISTS}/train.txt" "${LISTS}/val.txt"

# every_n_train_steps must be one epoch of OPTIMISER steps, or the periodic saves stop
# landing on epoch boundaries and the resume helper has nothing clean to restart from.
TRAIN_N=$(wc -l < "${LISTS}/train.txt")
echo
echo "SD-3.5 teacher set ready. Train with:"
echo "  bash scripts_rebuttal/run_sd35.sh"
echo
echo "Check configs/pretune_1M_sd35.yaml: with ${TRAIN_N} training images at global batch 192"
echo "one epoch is $(( (TRAIN_N + 191) / 192 )) optimiser steps (the file ships 5160)."
