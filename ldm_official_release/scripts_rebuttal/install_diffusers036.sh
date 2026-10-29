#!/bin/bash -l
# Install diffusers 0.36 into a directory NEXT TO the venv, for the FLUX.2 arm only.
#
#   bash scripts_rebuttal/install_diffusers036.sh          # -> $LDM_ROOT/pylibs/diffusers036
#
# WHY NOT JUST UPGRADE THE VENV. setup_env.sh pins diffusers 0.30.3, which every other arm
# has been run against; AutoencoderKLFlux2 only exists from 0.36. Upgrading in place would
# silently move the library under the ft-mse, SD-XL and SD-3.5 arms too. An overlay on
# PYTHONPATH is version-switchable per run and leaves the venv exactly as setup_env.sh
# built it:
#
#   EXTRA_PYTHONPATH=$LDM_ROOT/pylibs/diffusers036 bash scripts_rebuttal/run_flux2.sh
#
# scripts/env.sh reads EXTRA_PYTHONPATH and prepends it; run_flux2.sh sets it for you.
#
# --no-deps is deliberate: 0.36's own pins would drag in a newer huggingface_hub /
# transformers and defeat the point. The VAE classes need only torch, numpy and
# huggingface_hub, which the venv already has. Nothing else in 0.36 is used here.
#
# REQUIREMENTS. 0.36 is the last release supporting python 3.9 (setup_env.sh's cu121
# variant), and it touches torch.xpu at import, so it needs torch >= 2.4. Both setup_env.sh
# variants are newer (2.4.1+cu121 / 2.8.0+cu128), so either is fine. It will NOT work on an
# old torch 1.x environment -- the import dies on `torch.xpu`.
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
DST="${1:-${LDM_ROOT}/pylibs/diffusers036}"

if [ -d "${DST}/diffusers" ]; then
    echo "already installed: ${DST}"
else
    mkdir -p "${DST}"
    pip install --no-cache-dir --no-deps --target "${DST}" "diffusers==0.36.0"
fi

# Prove it loads AND that the class we need is there, in the venv this repo actually uses.
# A bare `pip show` would not catch a torch too old for 0.36's import.
PYTHONPATH="${DST}:${PYTHONPATH:-}" python - <<'PY'
import torch, diffusers
ok = hasattr(diffusers, "AutoencoderKLFlux2")
print("torch {}  diffusers {}  AutoencoderKLFlux2: {}".format(
    torch.__version__, diffusers.__version__, ok))
raise SystemExit(0 if ok else "diffusers loaded but has no AutoencoderKLFlux2")
PY

echo
echo "Use it with:  EXTRA_PYTHONPATH=${DST}"
echo "run_flux2.sh and prepare_flux2_teacher.sh default to exactly that path."
