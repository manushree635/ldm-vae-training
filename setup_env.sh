#!/bin/bash -l
# Build the official LDM (CompVis/latent-diffusion) training environment.
#
# Follows latent-diffusion/environment.yaml, with the CUDA/PyTorch pins raised
# from pytorch 1.7 + cudatoolkit 11.0 to pytorch 1.11.0 + cu113 -- exactly the
# bump CompVis themselves made in stable-diffusion's environment.yaml. cu113
# is required for sm_86 (A40 / A6000); the 1.7/cu110 pin only reaches sm_80.
# Every other pin (pytorch-lightning 1.4.2, omegaconf 2.1.1, einops 0.3.0,
# albumentations 0.4.3, ...) is upstream's.
set -euo pipefail

ROOT="/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official"
VENV="${ROOT}/venv_ldm"

module load python3/3.8.16
module load cuda/11.3

# NFS keeps .nfs* handles alive after an interrupted install, which makes a
# plain rm -rf fail mid-directory; rename first, then delete the renamed tree.
if [[ -e "${VENV}" ]]; then
    stale="${VENV}.stale.$$"
    mv "${VENV}" "${stale}"
    rm -rf "${stale}" || echo "warning: could not fully remove ${stale}"
fi
python3 -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade "pip==23.3.2" "setuptools==59.8.0" "wheel==0.37.1"

# --- PyTorch (CompVis/stable-diffusion pins) ---------------------------------
pip install --no-cache-dir \
    torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# --- environment.yaml pip section -------------------------------------------
pip install --no-cache-dir \
    "numpy==1.23.5" \
    "albumentations==0.4.3" \
    "opencv-python-headless==4.5.5.64" \
    "imageio==2.9.0" \
    "imageio-ffmpeg==0.4.2" \
    "pytorch-lightning==1.4.2" \
    "torchmetrics==0.6.0" \
    "omegaconf==2.1.1" \
    "test-tube>=0.7.5" \
    "einops==0.3.0" \
    "torch-fidelity==0.3.0" \
    "transformers==4.19.2" \
    "kornia==0.6.0" \
    "protobuf==3.20.3" \
    "tensorboard==2.11.2" \
    "pudb==2019.2" \
    "ftfy" "regex" "tqdm" "Pillow==9.5.0" "safetensors" \
    "urllib3<2"   # SCC's python3/3.8.16 links OpenSSL 1.0.2; urllib3 v2 needs 1.1.1+

# --- the three repos, as -e installs (environment.yaml does the same) --------
pip install --no-cache-dir --no-deps -e "${ROOT}/taming-transformers"
pip install --no-cache-dir --no-deps -e "${ROOT}/CLIP"
pip install --no-cache-dir --no-deps -e "${ROOT}/latent-diffusion"

python - <<'PY'
import torch, pytorch_lightning as pl, numpy, albumentations, omegaconf, einops
import taming, clip, ldm
print("torch              ", torch.__version__)
print("pytorch_lightning  ", pl.__version__)
print("numpy              ", numpy.__version__)
print("omegaconf          ", omegaconf.__version__)
print("einops             ", einops.__version__)
print("cuda archs         ", torch.cuda.get_arch_list())
PY
echo "OK: ${VENV}"
