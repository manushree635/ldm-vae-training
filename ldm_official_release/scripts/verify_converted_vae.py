"""Prove the diffusers -> LDM VAE conversion is exact, from the LDM side.

Instantiates LDM's own AutoencoderKL from the training YAML's ddconfig, loads
the converted checkpoint, replays the deterministic forward pass recorded by
convert_diffusers_vae_to_ldm.py, and compares against the diffusers reference.

After `source scripts/env.sh`:
    python scripts/verify_converted_vae.py \
        --ckpt models/sd15_vae_ldm.ckpt --config configs/autoencoder_kl_32x32x4_ftmse_20k.yaml
"""

import argparse
import sys

import torch
from omegaconf import OmegaConf

from ldm.models.autoencoder import AutoencoderKL

TOL = 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", default=None)
    args = ap.parse_args()
    ref_path = args.reference or args.ckpt.rsplit(".", 1)[0] + "_reference.pt"

    cfg = OmegaConf.load(args.config)
    params = cfg.model.params
    model = AutoencoderKL(ddconfig=params.ddconfig,
                          lossconfig={"target": "torch.nn.Identity"},
                          embed_dim=params.embed_dim).eval()

    sd = torch.load(args.ckpt, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("loss.")]
    print(f"missing (outside loss.*): {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print("  missing:", missing[:10])
    if unexpected:
        print("  unexpected:", unexpected[:10])

    ref = torch.load(ref_path, map_location="cpu")
    print(f"reference source: {ref['source']}")
    with torch.no_grad():
        posterior = model.encode(ref["input"])
        z = posterior.mode()
        recon = model.decode(z)

    d_latent = (z - ref["latent_mode"]).abs().max().item()
    d_recon = (recon - ref["recon"]).abs().max().item()
    print(f"max |latent delta| = {d_latent:.3e}")
    print(f"max |recon  delta| = {d_recon:.3e}")

    ok = not missing and not unexpected and d_latent < TOL and d_recon < TOL
    print("RESULT:", "PASS - LDM model is numerically identical to the diffusers VAE"
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
