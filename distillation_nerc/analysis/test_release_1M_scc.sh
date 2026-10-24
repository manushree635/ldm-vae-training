#!/bin/bash -l
# End-to-end test of ldm_official_release on the 1M path, as a newcomer would run it.
#
#   qsub analysis/test_release_1M_scc.sh
#
# Everything happens in node-local $TMPDIR on a FRESH copy of the repo, so:
#   - the real ldm_official_release/ is never written to (prepare_data_1M.sh rewrites
#     filelists/split_0_1M/, which would otherwise clobber the shipped lists)
#   - the 6 GB venv and the test data do not touch /projectnb, which is at 99.9% quota
#
# It runs the exact chain the README tells a newcomer to run, on 200 images instead of a
# million, and reports pass/fail per step. The point is to catch the failures only a real
# run surfaces -- a pin that no longer installs, a path assumption inside a prep script --
# not to produce a usable model.
#$ -P ivc-ml
#$ -pe omp 8
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l h_rt=4:00:00
#$ -N test_rel_1M
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/distillation_nerc/logs
set -uo pipefail
SRC=/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official_release
IMGS=/projectnb/ivc-ml/mvasu6/LAION_FACE/split_0
N_IMG=${N_IMG:-200}
W=${TMPDIR:-/scratch/$USER}/reltest
echo "host=$(hostname)  work=$W"; nvidia-smi --query-gpu=name --format=csv,noheader || true
df -h "${TMPDIR:-/scratch}" | tail -1

step() { echo; echo "######## $* ########"; }
FAIL=0
run() { local name="$1"; shift; if "$@"; then echo "PASS  $name"; else echo "FAIL  $name (exit $?)"; FAIL=1; fi; }

step "0  fresh copy of the repo (excluding venv/models/logs, as a git clone would be)"
mkdir -p "$W"
# LEADING SLASHES MATTER: a bare --exclude models matches at any depth and strips
# latent-diffusion/ldm/models/, which makes `import ldm.models` fail inside the copy.
rsync -a --exclude /venv --exclude /models --exclude /logs --exclude '__pycache__' "$SRC/" "$W/repo/"
mkdir -p "$W/repo/logs" "$W/data/split_0"
export LDM_ROOT="$W/repo" LDM_DATA="$W/data" LDM_VENV="$W/repo/venv"
# 200 images is enough to exercise every script; the teacher pass is the only slow part
(cd "$IMGS" && ls -1 | sort | head -n "$N_IMG") | while read -r f; do
    ln -sf "$IMGS/$f" "$W/data/split_0/$f"; done
echo "images staged: $(ls "$W/data/split_0" | wc -l)"

step "1  setup_env.sh  (builds the venv from pinned versions)"
run setup_env bash "$W/repo/setup_env.sh"
[ -x "$LDM_VENV/bin/python" ] || { echo "FATAL: no venv, cannot continue"; exit 1; }
"$LDM_VENV/bin/python" -c "import torch,pytorch_lightning as pl;print('torch',torch.__version__,'pl',pl.__version__,'cuda',torch.cuda.is_available())"

step "2  prepare_weights.sh  (ftmse_vae_ldm_with_disc.ckpt)"
run prepare_weights bash "$W/repo/scripts/prepare_weights.sh"
ls -la "$LDM_ROOT/models/" 2>/dev/null | head -5

step "3  prepare_data_1M.sh  (teacher recons + file lists + latent cache)"
# VAL_N defaults to 2000, which would consume the whole 200-image test set
export VAL_N=20
run prepare_data_1M bash "$W/repo/scripts/prepare_data_1M.sh"
echo "--- artefacts the 1M config reads:"
for p in "$LDM_DATA/teacher_recons_512_png" "$LDM_ROOT/filelists/split_0_1M/train.txt" \
         "$LDM_ROOT/filelists/split_0_1M/val.txt" \
         "$LDM_DATA/latent_cache/split_0_1M_ftmse_train.npy" \
         "$LDM_DATA/latent_cache/split_0_1M_ftmse_val.npy"; do
    if [ -e "$p" ]; then echo "  OK   $p  ($(du -sh "$p" 2>/dev/null | cut -f1))"; else echo "  MISS $p"; FAIL=1; fi
done

step "4  12 real training steps on configs/pretune_1M.yaml"
# val_n defaults to 2000 but we only have N_IMG images, so shrink the split for the test
cd "$LDM_ROOT"
run smoke env BATCH=2 CONFIG=configs/pretune_1M.yaml bash "$LDM_ROOT/scripts/smoke_test.sh"

echo
if [ "$FAIL" -eq 0 ]; then echo "RELEASE 1M TEST: ALL STEPS PASSED"; else echo "RELEASE 1M TEST: SOME STEPS FAILED (see above)"; fi
exit "$FAIL"
