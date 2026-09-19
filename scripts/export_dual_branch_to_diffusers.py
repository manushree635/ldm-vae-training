"""Export a dual-branch LDM checkpoint (autoencoder_dualbranch.py) for the diffusers scorers.

A dual-branch decoder has two output tails and no valid single-tail diffusers form, so this
writes a DIRECTORY that ldm_ftmse.dual_branch_diffusers knows how to load:

    <out>/branch1/          ordinary diffusers VAE (branch 1 keeps the original decoder keys)
    <out>/dual_branch.pt    branch 2's tensors, already in diffusers spelling, + the spec

There is deliberately no top-level config.json: a scorer that has not been taught about dual
arms will raise instead of quietly scoring branch 1 on its own.

Branch 2's keys are renamed onto branch 1's ldm spelling and pushed through the EXISTING
convert() in export_ldm_vae_to_diffusers.py, so there is only one key map in the project.

Run under the release venv (setup_env.sh installs diffusers).

    python scripts/export_dual_branch_to_diffusers.py --ckpt logs/<run>/checkpoints/last.ckpt \
        --config configs/..._dual_replace_20k.yaml --out models/dualrep_ep000024_ema --use-ema
"""
import argparse
import os
import sys

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from export_ldm_vae_to_diffusers import build, load_ldm_state_dict, swap_in_ema, convert  # noqa: E402

B2 = "decoder_b2."


def spec_from_config(path, n_blocks_ckpt):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    mp = cfg["model"]["params"]
    lp = mp["lossconfig"]["params"]
    match = str(lp.get("steerable_match_mode", "upsample")).strip().lower()
    spec = {
        "mode": str(mp["dual_branch_mode"]),
        "split_level": int(mp.get("dual_branch_split_level", 3)),
        "blocks": int(mp.get("dual_branch_blocks", 3)),
        "pad": int(mp.get("dual_branch_pad", 32)),
        "order": int(lp.get("steerable_order", 3)),
        "pyr_scale": 2 if match == "upsample" else 1,
    }
    if spec["blocks"] != n_blocks_ckpt:
        raise SystemExit("config says dual_branch_blocks={} but the checkpoint carries {} "
                         "branch-2 blocks".format(spec["blocks"], n_blocks_ckpt))
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True, help="the training yaml (for the split spec)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-ema", action="store_true")
    a = ap.parse_args()

    sd = load_ldm_state_dict(a.ckpt)
    if not any(k.startswith(B2) for k in sd):
        raise SystemExit("{} has no decoder_b2.* keys -- use export_ldm_vae_to_diffusers.py".format(a.ckpt))
    if a.use_ema:
        sd = swap_in_ema(sd)

    b2 = {k[len(B2):]: v for k, v in sd.items() if k.startswith(B2)}
    base = {k: v for k, v in sd.items() if not k.startswith(B2)}
    n_blocks = len({k.split(".")[1] for k in b2 if k.startswith("block.")})
    spec = spec_from_config(a.config, n_blocks)

    # Branch 1 = the ordinary decoder keys: the existing exporter handles it untouched.
    os.makedirs(a.out, exist_ok=True)
    vae = build(base)
    b1_dir = os.path.join(a.out, "branch1")
    vae.save_pretrained(b1_dir)
    print("branch 1 -> {}".format(b1_dir))

    # Branch 2: rename onto branch 1's ldm spelling, then reuse the SAME convert().
    keep = len(vae.decoder.up_blocks[-1].resnets) - n_blocks
    renamed = {}
    for k, v in b2.items():
        if k.startswith("block."):
            j = int(k.split(".")[1])
            renamed["decoder.up.0.block.{}.{}".format(keep + j, k.split(".", 2)[2])] = v
        elif k.startswith("norm_out.") or k.startswith("conv_out."):
            renamed["decoder." + k] = v
        elif k.startswith("attn."):
            raise SystemExit("branch-2 attention is not supported by the diffusers export")
        else:
            raise SystemExit("unexpected branch-2 key {!r}".format(k))
    tail = convert(renamed, list(vae.state_dict().keys()))
    torch.save({"spec": spec, "tail": tail}, os.path.join(a.out, "dual_branch.pt"))
    print("branch 2 -> {}/dual_branch.pt ({} tensors)".format(a.out, len(tail)))
    print("spec {}".format(spec))
    print("NOTE: load with ldm_ftmse.dual_branch_diffusers.load('{}'); plain "
          "from_pretrained on this dir will (deliberately) fail.".format(a.out))


if __name__ == "__main__":
    main()
