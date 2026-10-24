#!/bin/bash -l
# Build every initial checkpoint the configs start from, in LDM key layout:
#
#   models/sd15_vae_ldm.ckpt               SD-1.5 VAE             (configs 01, 02)
#   models/sd15_vae_ldm_with_disc.ckpt     + kl-f8 discriminator
#   models/ftmse_vae_ldm.ckpt              sd-vae-ft-mse
#   models/ftmse_vae_ldm_with_disc.ckpt    + kl-f8 discriminator  (configs 03-11)
#   models/ftema_vae_ldm_with_disc.ckpt    sd-vae-ft-ema + disc   (only with WITH_FTEMA=1)
#
#   bash scripts/prepare_weights.sh         # CPU only, ~10 min, ~2 GB of downloads
#
# Why a discriminator is grafted on: the GAN configs set disc_start to ~1 epoch,
# which is only sane with a PRETRAINED discriminator -- a randomly initialised one
# gives meaningless gradients for thousands of steps (the reason upstream defaults
# to disc_start=50001). CompVis' kl-f8.zip is a full training checkpoint and still
# carries its discriminator (step 246803), so it is lifted from there.
set -euo pipefail
export LDM_ROOT="${LDM_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# shellcheck disable=SC1091
source "${LDM_ROOT}/scripts/env.sh"
cd "${LDM_ROOT}"
M="${LDM_ROOT}/models"; mkdir -p "${M}"
: "${KLF8_URL:=https://ommer-lab.com/files/latent-diffusion/kl-f8.zip}"
: "${WITH_FTEMA:=0}"

if [ ! -f "${M}/kl-f8/model.ckpt" ]; then
    echo "### kl-f8 (for its discriminator)"
    mkdir -p "${M}/kl-f8"
    # -c resumes a partial download; the zip is ~1 GB.
    wget -c -O "${M}/kl-f8/kl-f8.zip" "${KLF8_URL}"
    unzip -o "${M}/kl-f8/kl-f8.zip" -d "${M}/kl-f8"
    rm -f "${M}/kl-f8/kl-f8.zip"
fi

convert() {  # name  hf_model  subfolder
    local name="$1" model="$2" sub="$3"
    if [ ! -f "${M}/${name}_vae_ldm.ckpt" ]; then
        echo "### ${model} -> ${name}_vae_ldm.ckpt"
        python scripts/convert_diffusers_vae_to_ldm.py --model "${model}" --subfolder "${sub}" \
            --out "${M}/${name}_vae_ldm.ckpt"
    fi
    if [ ! -f "${M}/${name}_vae_ldm_with_disc.ckpt" ]; then
        python scripts/merge_disc_into_ckpt.py --vae "${M}/${name}_vae_ldm.ckpt" \
            --disc "${M}/kl-f8/model.ckpt" --out "${M}/${name}_vae_ldm_with_disc.ckpt"
    fi
}
convert sd15  stable-diffusion-v1-5/stable-diffusion-v1-5 vae
convert ftmse stabilityai/sd-vae-ft-mse ""
[ "${WITH_FTEMA}" = "1" ] && convert ftema stabilityai/sd-vae-ft-ema ""

# The conversion is a hand-written key map; prove it from the LDM side rather than
# trusting it: replay the reference forward pass recorded at conversion time.
for name in sd15 ftmse; do
    echo "### verify ${name}"
    python scripts/verify_converted_vae.py --ckpt "${M}/${name}_vae_ldm.ckpt" \
        --config "${LDM_ROOT}/configs/pretune_1M.yaml"
done
ls -lh "${M}"/*.ckpt
