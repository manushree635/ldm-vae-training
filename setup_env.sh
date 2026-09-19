#!/bin/bash -l
# Build the one environment everything in this repo runs in: <repo>/venv.
#
#   bash setup_env.sh                 # torch 2.4.1+cu121 -- sm_50..sm_90 (V100 ... A100/A40/L40S/H100/H200)
#   TORCH_VARIANT=cu128 bash setup_env.sh    # torch 2.8.0+cu128 -- adds sm_120 (RTX PRO 6000 / Blackwell)
#
# ~10 minutes, ~6 GB. On BU SCC run it on a COMPUTE node (the login-node reaper
# kills the torch unpack):   qsub -P <project> -cwd -l h_rt=2:00:00 -pe omp 4 setup_env.sh
#
# Why these pins:
#   pytorch-lightning 1.9.5 -- the LAST release that supports multiple optimizers
#     under AUTOMATIC optimisation. The GAN loop here uses `optimizer_idx`
#     throughout; PL 2.x removed that path entirely.
#   albumentations 0.4.3 -- upstream LDM's pin. ImageNetSR's crop/resize geometry
#     is what the teacher reconstructions are aligned to, so a bump here is a
#     silent correctness risk, not housekeeping.
#   numpy < 2 -- albumentations 0.4.3 and imgaug predate the numpy 2 ABI.
#   diffusers -- only for checkpoint conversion/export and teacher precompute;
#     training itself never imports it.
set -euo pipefail

ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}"
[ -f "${ROOT}/latent-diffusion/main.py" ] || { echo "run from the repo root (got ROOT=${ROOT})" >&2; exit 1; }
VENV="${LDM_VENV:-${ROOT}/venv}"
: "${TORCH_VARIANT:=cu121}"

case "${TORCH_VARIANT}" in
    cu121) TORCH="torch==2.4.1+cu121 torchvision==0.19.1+cu121"; PYMOD="python3/3.9.9";   CUDAMOD="cuda/12.2" ;;
    cu128) TORCH="torch==2.8.0+cu128 torchvision==0.23.0+cu128"; PYMOD="python3/3.10.12"; CUDAMOD="cuda/12.8" ;;
    *) echo "TORCH_VARIANT must be cu121 or cu128" >&2; exit 1 ;;
esac

# Environment-Modules clusters only; elsewhere any python 3.9/3.10 on PATH is used.
if command -v module >/dev/null 2>&1; then
    module load "${LDM_PY_MODULE:-${PYMOD}}" || true
    module load "${LDM_CUDA_MODULE:-${CUDAMOD}}" || true
fi
python3 -c 'import sys; assert (3, 8) <= sys.version_info[:2] <= (3, 10), "PL 1.9 needs python 3.8-3.10, got " + sys.version'

if [[ -e "${VENV}" ]]; then
    echo "${VENV} already exists -- remove it first if you want a rebuild" >&2; exit 1
fi
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip wheel
# Recorded so scripts/env.sh loads the same modules this venv was built against.
printf 'LDM_PY_MODULE="%s"\nLDM_CUDA_MODULE="%s"\n' "${LDM_PY_MODULE:-${PYMOD}}" "${LDM_CUDA_MODULE:-${CUDAMOD}}" > "${VENV}/ldm_modules.sh"

# shellcheck disable=SC2086
pip install --no-cache-dir ${TORCH} --extra-index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"

pip install --no-cache-dir \
    "numpy==1.26.4" \
    "pytorch-lightning==1.9.5" \
    "torchmetrics==0.11.4" \
    "omegaconf==2.3.0" \
    "einops==0.8.0" \
    "opencv-python-headless==4.10.0.84" \
    "imageio==2.34.2" \
    "imageio-ffmpeg==0.5.1" \
    "torch-fidelity==0.3.0" \
    "transformers==4.44.2" \
    "diffusers==0.30.3" \
    "kornia==0.7.3" \
    "protobuf==4.25.4" \
    "tensorboard==2.17.1" \
    "Pillow==10.4.0" \
    "wandb==0.17.9" \
    "ftfy" "regex" "tqdm" "safetensors" "pandas" "scipy" \
    "scikit-image==0.22.0" \
    "urllib3<2"    # BU SCC's python3 links OpenSSL 1.0.2; urllib3 v2 needs 1.1.1+

# setuptools 69.x is REQUIRED for the next two: albumentations 0.4.3's setup.py
# imports pkg_resources, which setuptools >= 81 no longer ships.
pip install --no-cache-dir "setuptools==69.5.1"
pip install --no-cache-dir --no-build-isolation --no-deps "albumentations==0.4.3"
pip install --no-cache-dir --no-build-isolation --no-deps "imgaug==0.2.6"

# latent-diffusion/ldm and taming-transformers/taming have no __init__.py, so
# `pip install -e` writes an empty finder and `import ldm` still fails. They are
# put on PYTHONPATH by scripts/env.sh instead -- nothing to install here.

export PYTHONPATH="${ROOT}:${ROOT}/latent-diffusion:${ROOT}/taming-transformers"
python - <<'PY'
import torch, pytorch_lightning as pl, omegaconf, diffusers, albumentations
print(f"torch {torch.__version__}  PL {pl.__version__}  omegaconf {omegaconf.__version__}  "
      f"diffusers {diffusers.__version__}  albumentations {albumentations.__version__}")
# get_arch_list() is [] without a GPU (login node); that is not an error.
print("arch list:", torch.cuda.get_arch_list() or "(no GPU visible here)")
import ldm.models.autoencoder, taming.modules.losses.lpips, ldm_ftmse.autoencoder_steerable
print("ldm / taming / ldm_ftmse importable: OK")
PY
echo "done. next:  source scripts/env.sh"
