#!/bin/bash -l
# From one folder of images to everything the 20k configs read. Needs 1 GPU for
# the two teacher passes (~20 min per 20k images on an A40); the rest is CPU.
#
#   SRC_DIR=/path/to/my/images bash scripts/prepare_data.sh
#   qsub -P <project> -l gpus=1 -l gpu_c=7.0 -v SRC_DIR=/path/to/my/images scripts/prepare_data.sh
#
# Produces (names are the ones the configs already use -- keep them):
#
#   $LDM_DATA/subset_20k                                   your images (symlink to SRC_DIR)
#   $LDM_DATA/subset_20k_rcrop                             one 512x512 crop per image
#   $LDM_DATA/subset_20k_rcrop_teacher_recons_512_png      teacher recon of each crop      (configs 03-08, 10)
#   $LDM_DATA/subset_20k_teacher_recons_512_png            teacher recon of each full image (config 09)
#   filelists/subset_20k/{train,val}.txt                   95/5 split, seed 42              (configs 01, 02)
#   filelists/subset_20k_rcrop/{train,val}.txt             the SAME split, mapped to the crops
#   filelists/nerc_subset_20k/{train,val}.txt              the same split again (config 09)
#
# Every step skips work that is already done, so it is safe to re-run.
#$ -cwd
#$ -pe omp 8
#$ -l h_rt=6:00:00
#$ -N ldm_prepare_data
#$ -j y
#$ -o logs/
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
cd "${LDM_ROOT}"
: "${TEACHER:=stabilityai/sd-vae-ft-mse}"
: "${WORKERS:=8}"
mkdir -p "${LDM_DATA}"

if [ ! -e "${LDM_DATA}/subset_20k" ]; then
    : "${SRC_DIR:?set SRC_DIR to your image folder (or put the images at ${LDM_DATA}/subset_20k)}"
    ln -s "$(cd "${SRC_DIR}" && pwd)" "${LDM_DATA}/subset_20k"
fi

echo "### 1/5 split"
python scripts/make_filelists.py --data_dir "${LDM_DATA}/subset_20k" --out_dir filelists/subset_20k
mkdir -p filelists/nerc_subset_20k
cp filelists/subset_20k/train.txt filelists/subset_20k/val.txt filelists/nerc_subset_20k/

echo "### 2/5 512x512 crops"
python scripts/make_rcrop_dataset.py --src_dir "${LDM_DATA}/subset_20k" \
    --out_dir "${LDM_DATA}/subset_20k_rcrop" --workers "${WORKERS}"

echo "### 3/5 crop file lists (same train/val membership as step 1 -- do NOT re-split)"
python scripts/make_rcrop_filelists.py --src_lists filelists/subset_20k \
    --rcrop_dir "${LDM_DATA}/subset_20k_rcrop" --out_dir filelists/subset_20k_rcrop

echo "### 4/5 teacher recons of the crops"
python scripts/precompute_teacher_recons.py --teacher "${TEACHER}" --num_workers "${WORKERS}" \
    --src_dir "${LDM_DATA}/subset_20k_rcrop" --out_dir "${LDM_DATA}/subset_20k_rcrop_teacher_recons_512_png"

echo "### 5/5 teacher recons of the full images (config 09 only)"
python scripts/precompute_teacher_recons.py --teacher "${TEACHER}" --num_workers "${WORKERS}" \
    --src_dir "${LDM_DATA}/subset_20k" --out_dir "${LDM_DATA}/subset_20k_teacher_recons_512_png"

echo "data ready under ${LDM_DATA}"
