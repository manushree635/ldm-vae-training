"""Convert an LDM AutoencoderKL checkpoint into a diffusers AutoencoderKL directory.

The exact inverse of convert_diffusers_vae_to_ldm.py, so that a checkpoint trained
by latent-diffusion can be scored by tooling that speaks diffusers
(vae_bench loads every arm with AutoencoderKL.from_pretrained).

Correctness is checked, not asserted: --self-test exports the very checkpoint the
import produced and requires the result to be bit-identical to the diffusers VAE
it originally came from. Import followed by export must be the identity.

Needs diffusers, which setup_env.sh installs.

    python scripts/export_ldm_vae_to_diffusers.py --self-test
    python scripts/export_ldm_vae_to_diffusers.py \
        --ckpt logs/<run>/checkpoints/epoch=000000.ckpt --out models/ldm_ftema_epoch0_diffusers
"""

import argparse
import os
import sys

import torch

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def build_key_map():
    """(ldm_fragment, diffusers_fragment) pairs, applied ldm -> diffusers."""
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


def attn_suffix_map(template_keys):
    """Pick the diffusers attention spelling from the template model itself."""
    joined = "\n".join(template_keys)
    if "to_q." in joined:
        return {"norm.": "group_norm.", "q.": "to_q.", "k.": "to_k.",
                "v.": "to_v.", "proj_out.": "to_out.0."}
    if "query." in joined:
        return {"norm.": "group_norm.", "q.": "query.", "k.": "key.",
                "v.": "value.", "proj_out.": "proj_attn."}
    raise RuntimeError("could not detect the diffusers attention naming")


def convert(ldm_sd, template_keys):
    key_map = build_key_map()
    amap = attn_suffix_map(template_keys)
    linear_attn = "to_out.0.weight" in "\n".join(template_keys) or "proj_attn.weight" in "\n".join(template_keys)

    out = {}
    for k, v in ldm_sd.items():
        if k.startswith("loss.") or k.startswith("model_ema."):
            # loss.*      LPIPS + discriminator, not part of the VAE
            # model_ema.* LitEma shadow buffers; --use-ema folds them in first,
            #             and either way they are not diffusers parameters
            continue
        new_k = k
        if "attn_1." in new_k:
            head, _, tail = new_k.partition("attn_1.")
            for ldm_frag, hf_frag in amap.items():
                if tail.startswith(ldm_frag):
                    tail = hf_frag + tail[len(ldm_frag):]
                    break
            new_k = head + "attn_1." + tail
            # LDM stores attention projections as 1x1 convs; diffusers as Linear.
            if linear_attn and new_k.endswith(".weight") and v.ndim == 4:
                v = v.squeeze(-1).squeeze(-1)
        for ldm_frag, hf_frag in key_map:
            new_k = new_k.replace(ldm_frag, hf_frag)
        out[new_k] = v
    return out


def load_ldm_state_dict(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["state_dict"] if "state_dict" in ck else ck


def swap_in_ema(sd):
    """Replace trained params with their EMA shadow copies.

    LitEma stores each shadow as a buffer named `model_ema.<param name with the
    dots stripped>` (dots are illegal in buffer names). Only parameters with
    requires_grad get a shadow, so a frozen encoder simply has none and keeps
    its raw -- and identical -- weights.
    """
    shadows = {k[len("model_ema."):]: v for k, v in sd.items()
               if k.startswith("model_ema.") and k.split(".")[-1] not in ("decay", "num_updates")}
    if not shadows:
        raise SystemExit("--use-ema given but this checkpoint has no model_ema.* buffers")
    out, swapped = {}, 0
    for k, v in sd.items():
        if k.startswith("model_ema."):
            continue
        s_name = k.replace(".", "")
        if s_name in shadows:
            out[k] = shadows[s_name]; swapped += 1
        else:
            out[k] = v
    print(f"  EMA: swapped {swapped} of {len(out)} tensors "
          f"({len(shadows)} shadows available)")
    return out


def build(ldm_sd, dtype=torch.float32):
    from diffusers import AutoencoderKL
    template = AutoencoderKL.from_pretrained(SD15, subfolder="vae")
    converted = convert(ldm_sd, list(template.state_dict().keys()))
    missing, unexpected = template.load_state_dict(converted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"key mismatch — missing={missing[:6]} unexpected={unexpected[:6]}")
    return template.to(dtype)


def self_test():
    """import -> export must be the identity on the SD-1.5 VAE."""
    from diffusers import AutoencoderKL
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt = os.path.join(here, "models", "sd15_vae_ldm.ckpt")
    print(f"round-tripping {ckpt}")
    exported = build(load_ldm_state_dict(ckpt))
    original = AutoencoderKL.from_pretrained(SD15, subfolder="vae")
    a, b = exported.state_dict(), original.state_dict()
    assert set(a) == set(b), "key sets differ"
    worst, worst_k = 0.0, None
    for k in a:
        d = (a[k].float() - b[k].float()).abs().max().item()
        if d > worst:
            worst, worst_k = d, k
    print(f"  {len(a)} tensors compared")
    print(f"  max |delta| = {worst:.3e}" + (f"  at {worst_k}" if worst_k else ""))
    ok = worst == 0.0
    print("  RESULT:", "PASS — round trip is exact" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--use-ema", action="store_true",
                    help="export the EMA shadow weights instead of the trained ones")
    args = ap.parse_args()

    if args.self_test:
        rc = self_test()
        if not args.ckpt:
            return rc
        if rc:
            raise SystemExit("self-test failed; refusing to export")

    if not args.ckpt or not args.out:
        raise SystemExit("--ckpt and --out are required (or use --self-test alone)")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    if args.use_ema:
        sd = swap_in_ema(sd)
    model = build(sd)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"exported {args.ckpt}")
    print(f"  epoch={ck.get('epoch')} global_step={ck.get('global_step')} -> {args.out}")
    print(f"  files: {sorted(os.listdir(args.out))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
