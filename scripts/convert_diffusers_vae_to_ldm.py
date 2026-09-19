"""Convert a diffusers AutoencoderKL into LDM's `first_stage_model` layout.

LDM's AutoencoderKL predates diffusers and uses the original CompVis key names
(`encoder.down.0.block.0.norm1`, attention as 1x1 convs, up-blocks numbered in
the opposite order). This is the inverse of diffusers'
`convert_ldm_vae_checkpoint`, i.e. the `convert_vae_state_dict` direction from
diffusers' own scripts/convert_diffusers_to_original_stable_diffusion.py.

Needs diffusers, which setup_env.sh installs. It also writes a reference forward pass so that
verify_converted_vae.py can prove the conversion numerically from the LDM side.

    python scripts/convert_diffusers_vae_to_ldm.py \
        --model stable-diffusion-v1-5/stable-diffusion-v1-5 --subfolder vae \
        --out models/sd15_vae_ldm.ckpt
"""

import argparse
import os

import torch


def build_key_map():
    """(ldm_fragment, diffusers_fragment) pairs, applied diffusers -> ldm."""
    m = [
        ("nin_shortcut", "conv_shortcut"),
        ("norm_out", "conv_norm_out"),
        ("mid.attn_1.", "mid_block.attentions.0."),
    ]
    for i in range(4):
        for j in range(2):
            m.append((f"encoder.down.{i}.block.{j}.",
                      f"encoder.down_blocks.{i}.resnets.{j}."))
        if i < 3:
            m.append((f"down.{i}.downsample.", f"down_blocks.{i}.downsamplers.0."))
            m.append((f"up.{3 - i}.upsample.", f"up_blocks.{i}.upsamplers.0."))
        for j in range(3):
            m.append((f"decoder.up.{3 - i}.block.{j}.",
                      f"decoder.up_blocks.{i}.resnets.{j}."))
    for i in range(2):
        m.append((f"mid.block_{i + 1}.", f"mid_block.resnets.{i}."))
    return m


# diffusers renamed the attention projections partway through its life; accept
# both spellings so this works against old and new snapshots alike.
ATTN_MAPS = [
    [("norm.", "group_norm."), ("q.", "to_q."), ("k.", "to_k."),
     ("v.", "to_v."), ("proj_out.", "to_out.0.")],
    [("norm.", "group_norm."), ("q.", "query."), ("k.", "key."),
     ("v.", "value."), ("proj_out.", "proj_attn.")],
]


def convert(hf_sd):
    key_map = build_key_map()
    attn_keys = [k for k in hf_sd if "attentions" in k]
    attn_map = None
    for candidate in ATTN_MAPS:
        if any(frag in k for k in attn_keys for _, frag in candidate if frag != "group_norm."):
            attn_map = candidate
            break
    if attn_map is None:
        raise RuntimeError(f"unrecognised attention key naming: {attn_keys[:6]}")

    out = {}
    for k, v in hf_sd.items():
        new_k = k
        if "attentions" in k:
            for ldm_frag, hf_frag in attn_map:
                new_k = new_k.replace(hf_frag, ldm_frag)
        for ldm_frag, hf_frag in key_map:
            new_k = new_k.replace(hf_frag, ldm_frag)
        # LDM implements mid-block attention as 1x1 convs, diffusers as Linear.
        if "attn_1" in new_k and new_k.endswith(".weight") and v.ndim == 2:
            v = v.reshape(*v.shape, 1, 1)
        out[new_k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--subfolder", default="vae")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=None,
                    help="where to write the reference forward pass "
                         "(default: alongside --out as *_reference.pt)")
    args = ap.parse_args()

    from diffusers import AutoencoderKL

    kwargs = {"subfolder": args.subfolder} if args.subfolder else {}
    vae = AutoencoderKL.from_pretrained(args.model, **kwargs).eval()
    ldm_sd = convert(vae.state_dict())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": ldm_sd}, args.out)
    print(f"wrote {len(ldm_sd)} tensors -> {args.out}")

    # Reference pass, deterministic: encode to the posterior *mode* (no sampling)
    # and decode, so the LDM side can reproduce it exactly.
    ref_path = args.reference or os.path.splitext(args.out)[0] + "_reference.pt"
    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, 3, 256, 256, generator=g) * 2 - 1
    with torch.no_grad():
        moments = vae.quant_conv(vae.encoder(x))
        z = moments.chunk(2, dim=1)[0]           # mean == posterior mode
        recon = vae.decoder(vae.post_quant_conv(z))
    torch.save({"input": x, "latent_mode": z, "recon": recon,
                "source": f"{args.model}:{args.subfolder}"}, ref_path)
    print(f"wrote reference forward pass -> {ref_path}")
    print(f"  latent {tuple(z.shape)} recon {tuple(recon.shape)} "
          f"recon range [{recon.min():.4f}, {recon.max():.4f}]")


if __name__ == "__main__":
    main()
