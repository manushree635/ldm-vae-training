#!/bin/bash -l
# From one folder of images to everything the 1M configs read (pretune_1M,
# posttune_1M, 11_steerable_graded_1M, 12_dual_branch_1M).
#
#   SRC_DIR=/path/to/my/1M/images bash scripts_rebuttal/prepare_data_1M.sh
#   qsub -P <project> -l gpus=1 -l gpu_c=7.0 -v SRC_DIR=/path/to/images scripts_rebuttal/prepare_data_1M.sh
#
# Produces (names are the ones the 1M configs already use -- keep them):
#
#   $LDM_DATA/split_0                                 your images (symlink to SRC_DIR)
#   $LDM_DATA/teacher_recons_512_png                  teacher recon of each image, 512 PNG
#   filelists/split_0_1M/{train,val}.txt              2,000-image val split, teacher-intersected
#   $LDM_DATA/latent_cache/split_0_1M_ftmse_train.npy frozen-encoder latents
#   $LDM_DATA/latent_cache/split_0_1M_ftmse_val.npy
#
# WHY THIS IS SEPARATE FROM prepare_data.sh. That script builds the 20k set and
# stops; nothing in it produces the three artefacts above, so the 1M configs could
# not be run from a fresh checkout. It also differs in substance, not just scale:
#
#   no crops      the 20k path makes one 512x512 random crop per image and trains
#                 on those. The 1M path trains on the WHOLE frame squashed to
#                 256x256, paired with a teacher recon of the whole frame squashed
#                 to 512x512 (ldm_ftmse/data_squash.py). Do not feed it rcrops.
#   intersection  make_filelists_teacher.py keeps only images that HAVE a teacher
#                 recon. data_teacher.py's verify_teacher aborts the run on a
#                 missing one, and at this scale a handful always fail to decode.
#   latent cache  the encoder is frozen and the crop is deterministic, so its
#                 output is cached once instead of recomputed every epoch. This is
#                 bit-exact, not an approximation -- but it is keyed to the file
#                 list, so REBUILD IT if the lists change. A stale cache silently
#                 trains on the wrong latents; precompute_latents.py writes a
#                 fingerprint sidecar and data_teacher.py refuses a mismatch.
#
# COST. The teacher pass dominates: ~1 GPU-day for 1M images on an A40, and the
# PNGs are ~350 GB. The latent cache is 32 KB/image, so ~32 GB, and takes a couple
# of hours. Budget both before starting; every step skips work already done, so an
# interrupted run can simply be re-submitted.
#
# A DIFFERENT TEACHER. Set TEACHER to any diffusers AutoencoderKL to distil from
# something other than sd-vae-ft-mse, but then also point ckpt_path in the config
# at that VAE converted to LDM layout (scripts/convert_diffusers_vae_to_ldm.py),
# or the student starts from one model and is pulled towards another.
#$ -cwd
#$ -pe omp 8
#$ -l h_rt=24:00:00
#$ -N ldm_prepare_1M
#$ -j y
#$ -o logs/
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
cd "${LDM_ROOT}"
: "${TEACHER:=stabilityai/sd-vae-ft-mse}"
: "${WORKERS:=8}"
: "${VAL_N:=2000}"
: "${CONFIG:=configs/pretune_1M.yaml}"
mkdir -p "${LDM_DATA}" "${LDM_DATA}/latent_cache"

if [ ! -e "${LDM_DATA}/split_0" ]; then
    : "${SRC_DIR:?set SRC_DIR to your image folder (or put the images at ${LDM_DATA}/split_0)}"
    ln -s "$(cd "${SRC_DIR}" && pwd)" "${LDM_DATA}/split_0"
fi
echo "images: $(ls "${LDM_DATA}/split_0" | wc -l) under ${LDM_DATA}/split_0"

echo "### 1/3 teacher recons (the long one: ~1 GPU-day and ~350 GB for 1M images)"
python scripts/precompute_teacher_recons.py --teacher "${TEACHER}" --num_workers "${WORKERS}" \
    --src_dir "${LDM_DATA}/split_0" --out_dir "${LDM_DATA}/teacher_recons_512_png"

echo "### 2/3 file lists, keeping only images that have a teacher recon"
python scripts/make_filelists_teacher.py \
    --data_dir    "${LDM_DATA}/split_0" \
    --teacher_dir "${LDM_DATA}/teacher_recons_512_png" \
    --out_dir     filelists/split_0_1M --val_n "${VAL_N}"
wc -l filelists/split_0_1M/train.txt filelists/split_0_1M/val.txt

echo "### 3/3 frozen-encoder latent cache (keyed to the lists above)"
for split in train validation; do
    case "${split}" in train) out=train;; validation) out=val;; esac
    python scripts/precompute_latents.py --config "${CONFIG}" --split "${split}" \
        --out "${LDM_DATA}/latent_cache/split_0_1M_ftmse_${out}.npy" \
        --batch-size 32 --num-workers "${WORKERS}"
done

echo
echo "1M data ready. Train with:"
echo "  bash scripts/train.sh ${CONFIG}"
