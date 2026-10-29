#!/bin/bash -l
# Generate the FLUX.2 TEACHER output for the 1M set, then the file lists that go with it.
# prepare_sd35_teacher.sh with the model swapped: same images, same Resize((512,512)) squash,
# same posterior-mean round trip. Feeds configs/pretune_1M_flux2.yaml.
#
#   bash scripts_rebuttal/prepare_flux2_teacher.sh
#   qsub -P <project> -l gpus=4 -l gpu_c=8.0 -pe omp 16 scripts_rebuttal/prepare_flux2_teacher.sh
#
#   SRC_DIR=...   image folder, if $LDM_DATA/split_0 is not already there
#   NUM_SHARDS=   processes, one per granted GPU (default: as many GPUs as are visible)
#   BATCH=8  WORKERS=8  VAL_N=2000
#   VAE_REPO=     local VAE dir (default $LDM_ROOT/models/flux2_stock_vae, saved below)
#   HF_REPO=      where to fetch it from (default black-forest-labs/FLUX.2-dev, subfolder vae)
#
# Produces:
#   $LDM_ROOT/models/flux2_stock_vae                     the VAE itself (once)
#   $LDM_DATA/teacher_recons_512_png_flux2               one 512 PNG per image, ~350 GB
#   $LDM_ROOT/filelists/split_0_1M_flux2/{train,val}.txt teacher-intersected, val_n 2000
#
# TWO THINGS THIS NEEDS THAT THE SD-3.5 PASS DID NOT:
#
#   diffusers 0.36  AutoencoderKLFlux2 does not exist in the pinned 0.30.3. EXTRA_PYTHONPATH
#                   points at the overlay; scripts/env.sh prepends it, leaving the venv as
#                   setup_env.sh built it. Run install_diffusers036.sh once first.
#   HF licence      black-forest-labs/FLUX.2-dev is gated. `huggingface-cli login` with an
#                   account that has accepted it, or the fetch below 401s. Saving the VAE
#                   locally also means the sharded pass does not re-fetch it per process.
#
# WHY A SEPARATE TEACHER SET. The steerable term pulls the student towards what the teacher
# reconstructs at 512, and for this arm the teacher IS FLUX.2, so ft-mse's recons are the
# wrong target -- they would cap the term at a weaker VAE's output. A separate pass also
# drops a different handful of undecodable images, so the file lists are rebuilt rather than
# reused from split_0_1M.
#
# WHY NO LATENT CACHE STEP. prepare_data_1M.sh ends by caching the frozen encoder's output.
# That cache is ft-mse's 4-channel latents; this encoder is a different function with a
# 32-channel output, and this arm's model class has no cache path. Nothing to build.
#
# COST AND RESUMING. FLUX.2's VAE is the same f8 size as kl-f8 (84.0M), so budget roughly
# the ~1 GPU-day prepare_data_1M.sh quotes for 1M images, divided by NUM_SHARDS. That will
# exceed a wall clock on few GPUs, which is fine -- images whose PNG exists are skipped, so
# resubmit until the count stops moving, and shards re-split across the remaining work each
# time so they stay balanced.
#
# DISK. ~350 GB. Check before starting; a full filesystem mid-pass leaves truncated PNGs,
# which decode as errors at train time rather than being skipped.
#$ -cwd
#$ -pe omp 16
#$ -l h_rt=24:00:00
#$ -N flux2_teacher
#$ -j y
#$ -o logs/
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
export EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-${LDM_ROOT}/pylibs/diffusers036}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
cd "${LDM_ROOT}"

: "${VAE_REPO:=${LDM_ROOT}/models/flux2_stock_vae}"
: "${HF_REPO:=black-forest-labs/FLUX.2-dev}"
: "${BATCH:=8}"
: "${WORKERS:=8}"
: "${VAL_N:=2000}"
OUT="${LDM_DATA}/teacher_recons_512_png_flux2"
LISTS=filelists/split_0_1M_flux2
mkdir -p "${LDM_DATA}" "${OUT}" "${LISTS}" logs

[ -d "${EXTRA_PYTHONPATH}/diffusers" ] || {
    echo "FATAL: no diffusers overlay at ${EXTRA_PYTHONPATH}" >&2
    echo "       run scripts_rebuttal/install_diffusers036.sh first" >&2; exit 1; }

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

echo "### 1/3 the FLUX.2 VAE itself"
if [ -f "${VAE_REPO}/config.json" ]; then
    echo "have it: ${VAE_REPO}"
else
    echo "fetching ${HF_REPO}:vae into ${VAE_REPO} (gated -- huggingface-cli login first)"
    mkdir -p "${VAE_REPO}"
    python - "$HF_REPO" "$VAE_REPO" <<'PY'
import sys
import torch
from diffusers import AutoencoderKLFlux2
src, dst = sys.argv[1], sys.argv[2]
vae = AutoencoderKLFlux2.from_pretrained(src, subfolder="vae", torch_dtype=torch.float32)
vae.save_pretrained(dst)
c = vae.config
print("latent_channels {}  block_out_channels {}  stages {}".format(
    c.latent_channels, list(c.block_out_channels), len(c.down_block_types)))
print("saved -> {}".format(dst))
PY
fi

echo "### 2/3 teacher recons at 512 (the long one), ${NUM_SHARDS} shard(s) -> ${OUT}"
pids=()
for ((i = 0; i < NUM_SHARDS; i++)); do
    dev="${GRANTED[$i]}"
    echo "--- shard ${i}/${NUM_SHARDS} on GPU ${dev}"
    CUDA_VISIBLE_DEVICES="${dev}" python scripts/precompute_teacher_recons.py \
        --teacher "${VAE_REPO}" \
        --vae_class AutoencoderKLFlux2 \
        --src_dir "${LDM_DATA}/split_0" \
        --out_dir "${OUT}" \
        --size 512 --batch_size "${BATCH}" --num_workers "${WORKERS}" \
        --num_shards "${NUM_SHARDS}" --shard_index "${i}" \
        > "logs/flux2_teacher_shard${i}.log" 2>&1 &
    pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
tail -n 2 logs/flux2_teacher_shard*.log || true
HAVE=$(find "${OUT}" -maxdepth 1 -name '*.png' | wc -l)
echo "recons present: ${HAVE} / ${N_IMG}"
if [ "${status}" -ne 0 ]; then echo "FATAL: a shard failed -- see logs/flux2_teacher_shard*.log" >&2; exit 1; fi
if [ "${HAVE}" -lt "${N_IMG}" ]; then
    # Expected on the first submission: this is more than a wall clock's work.
    echo "INCOMPLETE: resubmit this script to continue (finished images are not redone)."
    echo "Stopping before the file lists, which must describe a COMPLETE teacher set."
    exit 0
fi

echo "### 3/3 file lists, keeping only images that have a FLUX.2 recon"
python scripts/make_filelists_teacher.py \
    --data_dir    "${LDM_DATA}/split_0" \
    --teacher_dir "${OUT}" \
    --out_dir     "${LISTS}" --val_n "${VAL_N}"
wc -l "${LISTS}/train.txt" "${LISTS}/val.txt"

# every_n_train_steps must be one epoch of OPTIMISER steps, or the periodic saves stop
# landing on epoch boundaries and the resume helper has nothing clean to restart from.
TRAIN_N=$(wc -l < "${LISTS}/train.txt")
echo
echo "FLUX.2 teacher set ready. Train with:"
echo "  bash scripts_rebuttal/run_flux2.sh"
echo
echo "Check configs/pretune_1M_flux2.yaml: with ${TRAIN_N} training images at global batch 192"
echo "one epoch is $(( (TRAIN_N + 191) / 192 )) optimiser steps (the file ships 5160)."
