from __future__ import annotations
"""Run a dual-branch decoder (autoencoder_dualbranch.py) inside a diffusers AutoencoderKL.

WHY THIS EXISTS. Every scorer in this project -- COCO rFID, the 512-crop round-trip bench,
TokBench -- loads arms with `AutoencoderKL.from_pretrained`. diffusers' decoder has ONE
output tail and cannot represent two, so a dual-branch checkpoint has no valid diffusers
form. Rather than reimplement the scorers, this rebuilds the second tail out of diffusers'
own modules and rebinds `vae.decoder.forward`.

The export (scripts/export_dual_branch_to_diffusers.py) writes

    <dir>/branch1/            an ordinary diffusers VAE -- branch 1 keeps the original
                              decoder keys, so it converts with the existing key map
    <dir>/dual_branch.pt      branch 2's tensors (already in diffusers spelling) + the spec

and deliberately writes NO top-level config.json, so `from_pretrained(<dir>)` raises instead
of silently scoring branch 1 alone in a scorer that has not been taught about dual arms.

`<dir>/branch1` remains loadable by itself, on purpose: in "replace" mode branch 1 is a
full-image L1+LPIPS tail, and scoring it alone is a useful diagnostic.
"""

import copy
import json
import os

import torch
import torch.nn as nn

from . import band_split

SIDECAR = "dual_branch.pt"
BRANCH1 = "branch1"


def is_dual(path) -> bool:
    return bool(path) and os.path.isfile(os.path.join(str(path), SIDECAR))


def branch1_dir(path) -> str:
    return os.path.join(str(path), BRANCH1)


def _tail_state_dict(tail_keys, keep):
    """Diffusers-spelled export keys -> submodule names on the tail module built below."""
    out = {}
    for k, v in tail_keys.items():
        if ".resnets." in k:
            head, _, rest = k.partition(".resnets.")
            idx, _, sub = rest.partition(".")
            out["resnets.{}.{}".format(int(idx) - keep, sub)] = v
        elif ".conv_norm_out." in k:
            out["conv_norm_out." + k.split(".conv_norm_out.", 1)[1]] = v
        elif ".conv_out." in k:
            out["conv_out." + k.split(".conv_out.", 1)[1]] = v
        else:
            raise KeyError("unexpected dual-branch tail key {!r}".format(k))
    return out


def attach(vae, path):
    """Give `vae` its second tail and the band-split decode. Returns `vae`."""
    blob = torch.load(os.path.join(str(path), SIDECAR), map_location="cpu")
    spec, tail_sd = blob["spec"], blob["tail"]

    dec = vae.decoder
    last = dec.up_blocks[-1]
    if getattr(last, "upsamplers", None):
        raise RuntimeError("the last up_block has upsamplers; the branch point assumes it does not")
    n_blocks = len(last.resnets)
    keep = n_blocks - int(spec["blocks"])
    if not 0 <= keep <= n_blocks:
        raise ValueError("dual spec blocks={} does not fit {} resnets".format(spec["blocks"], n_blocks))

    ref = dec.conv_out.weight
    tail = nn.Module()
    tail.resnets = nn.ModuleList([copy.deepcopy(r) for r in last.resnets[keep:]])
    tail.conv_norm_out = copy.deepcopy(dec.conv_norm_out)
    tail.conv_out = copy.deepcopy(dec.conv_out)
    tail.load_state_dict({k: v.to(ref.dtype) for k, v in _tail_state_dict(tail_sd, keep).items()},
                         strict=True)
    tail = tail.to(device=ref.device, dtype=ref.dtype).eval()
    for p in tail.parameters():
        p.requires_grad_(False)
    dec.dual_tail = tail            # a child of decoder, so `.to(fp32)` in a scorer covers it
    dec.dual_spec = dict(spec)

    act = dec.conv_act

    def forward(sample, latent_embeds=None):
        # diffusers.models.autoencoders.vae.Decoder.forward, split at the last up block.
        h = dec.conv_in(sample)
        h = dec.mid_block(h, latent_embeds)
        for up in dec.up_blocks[:-1]:
            h = up(h, latent_embeds)
        for r in last.resnets[:keep]:
            h = r(h, None)

        def run(resnets, norm, conv):
            y = h
            for r in resnets:
                y = r(y, None)
            y = norm(y) if latent_embeds is None else norm(y, latent_embeds)
            return conv(act(y))

        b1 = run(last.resnets[keep:], dec.conv_norm_out, dec.conv_out)
        b2 = run(tail.resnets, tail.conv_norm_out, tail.conv_out)
        out = band_split.mix(b1, b2, int(spec["split_level"]), int(spec["pad"]),
                             order=int(spec["order"]), pyr_scale=int(spec["pyr_scale"]))
        return out.to(b1.dtype)

    dec.forward = forward
    return vae


def load(path, **from_pretrained_kwargs):
    """`AutoencoderKL.from_pretrained`, dual-aware. Non-dual paths behave exactly as before."""
    from diffusers import AutoencoderKL
    if not is_dual(path):
        return AutoencoderKL.from_pretrained(str(path), **from_pretrained_kwargs)
    vae = AutoencoderKL.from_pretrained(branch1_dir(path), **from_pretrained_kwargs)
    return attach(vae, path)


def spec_of(path):
    blob = torch.load(os.path.join(str(path), SIDECAR), map_location="cpu")
    return blob["spec"]


def describe(path):
    return json.dumps(spec_of(path), sort_keys=True)


def patch_from_pretrained():
    """Make `AutoencoderKL.from_pretrained` dual-aware for the rest of this process.

    For scorers that call from_pretrained directly (evaluate_model_metrics.py) rather than
    through a loader function that can be wrapped. A non-dual path goes to the original
    classmethod untouched, so baselines/teachers loaded in the same process are unaffected.
    Idempotent.
    """
    from diffusers import AutoencoderKL
    if getattr(AutoencoderKL, "_dual_patched", False):
        return
    original = AutoencoderKL.from_pretrained.__func__

    def from_pretrained(cls, pretrained_model_name_or_path=None, *args, **kwargs):
        path = pretrained_model_name_or_path
        if path is not None and is_dual(path):
            kwargs.pop("subfolder", None)          # a dual export has no subfolder layout
            vae = original(cls, branch1_dir(path), *args, **kwargs)
            attach(vae, path)
            print("[dual] {} loaded as a dual-branch decoder {}".format(path, describe(path)), flush=True)
            return vae
        return original(cls, path, *args, **kwargs)

    AutoencoderKL.from_pretrained = classmethod(from_pretrained)
    AutoencoderKL._dual_patched = True
