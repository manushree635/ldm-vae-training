from __future__ import annotations
"""Decoder with TWO output tails, combined by a fixed steerable-band split.

THE IDEA. Every arm so far has one decoder output and relies on the loss alone to decide
which frequencies each term shapes. Here the last full-resolution block of the decoder is
duplicated, giving two tails that share everything before it:

    z -> post_quant_conv -> [ mid + up[3..1] + first blocks of up[0] ]  = trunk
                                     |                      |
                              branch 1 tail          branch 2 tail      (independent weights)
                                     |                      |
                             out = LP(b1) + HP(b2)   (band_split.mix)

LP keeps the steerable pyramid's levels >= split_level plus residual_lowpass; HP = 1 - LP
keeps residual_highpass and levels 0..split_level-1. With the default split_level=3 on the
512 loss pyramid, the crossover sits at ~23 cycles/image of the 256 output (transition
16 -> 32 cpi), i.e. branch 2 owns the octaves above the 32x32 latent's Nyquist, including
the 16-64 cpi bands where face identity lives.

BOTH BRANCHES START AS COPIES of the pretrained tail. Because mix() is written as
b2 + LP(b1 - b2), identical branches give out == b1 EXACTLY, so step 0 reproduces the
checkpoint being finetuned rather than perturbing it.

WHICH LOSS TRAINS WHICH TAIL is losses_dualbranch.LPIPSDualBranchSteerable's job; this
class only stashes (b1, b2) on the loss the way AutoencoderKLSteerableEMA stashes the
teacher, which covers all five self.loss(...) call sites including the EMA validation pass.

ORDERING TRAPS, both already met by autoencoder_probe.py:
  * decoder_b2 is built AFTER super().__init__(), so upstream's init_from_ckpt -- which
    runs inside AutoencoderKL.__init__ -- never sees it as a missing key. The flip side is
    that a DUAL checkpoint passed as ckpt_path would leave branch 2 as a copy of branch 1,
    so its decoder_b2.* tensors are loaded explicitly below.
  * LitEma only shadows parameters that had requires_grad when IT was built, so the EMA is
    rebuilt after the new tail exists. Otherwise --use-ema would export a branch 2 that
    never moved.
  * configure_optimizers lists modules explicitly upstream, so the new parameters are added
    to the existing group by hand.
"""

import copy

import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.model import nonlinearity
from ldm.modules.ema import LitEma

from . import band_split
from .autoencoder_nogan import AutoencoderKLSteerableEMANoGAN
from .autoencoder_steerable import AutoencoderKLSteerableEMA
from .losses_dualbranch import LPIPSDualBranchSteerable

_DUAL_MODES = ("split", "replace")


class DualBranchMixin:
    def __init__(self, *args, dual_branch_mode="replace", dual_branch_split_level=3,
                 dual_branch_blocks=3, dual_branch_pad=32, dual_branch_steer_all_bands=False, **kwargs):
        self.dual_branch_mode = str(dual_branch_mode).strip().lower()
        if self.dual_branch_mode not in _DUAL_MODES:
            raise ValueError("dual_branch_mode must be one of {}, got {!r}".format(
                _DUAL_MODES, dual_branch_mode))
        self.dual_branch_split_level = int(dual_branch_split_level)
        self.dual_branch_blocks = int(dual_branch_blocks)
        self.dual_branch_pad = int(dual_branch_pad)
        self.dual_branch_steer_all_bands = bool(dual_branch_steer_all_bands)

        super().__init__(*args, **kwargs)

        d = self.decoder
        if getattr(d, "give_pre_end", False):
            raise ValueError("dual-branch decoder needs give_pre_end=False")
        n_blocks = d.num_res_blocks + 1
        if not 0 <= self.dual_branch_blocks <= n_blocks:
            raise ValueError("dual_branch_blocks must be 0..{}, got {}".format(
                n_blocks, self.dual_branch_blocks))
        if not isinstance(self.loss, LPIPSDualBranchSteerable):
            raise ValueError(
                "dual-branch decoder needs lossconfig.target "
                "ldm_ftmse.losses_dualbranch.LPIPSDualBranchSteerable, got {}".format(
                    type(self.loss).__name__))

        # Branch 2 = a deep copy of branch 1's tail: the trailing blocks of the last
        # (full-resolution) up level, plus norm_out and conv_out.
        keep = n_blocks - self.dual_branch_blocks
        b2 = nn.Module()
        b2.block = nn.ModuleList([copy.deepcopy(b) for b in d.up[0].block[keep:]])
        b2.attn = nn.ModuleList([copy.deepcopy(a) for a in d.up[0].attn[keep:]]) \
            if len(d.up[0].attn) > 0 else nn.ModuleList()
        b2.norm_out = copy.deepcopy(d.norm_out)
        b2.conv_out = copy.deepcopy(d.conv_out)
        b2.requires_grad_(True)
        self.decoder_b2 = b2

        # A dual checkpoint handed to ckpt_path: init_from_ckpt ran before decoder_b2
        # existed, so branch 2 would silently be a copy of branch 1.
        ckpt_path = kwargs.get("ckpt_path")
        if ckpt_path:
            self._load_branch2_from_ckpt(ckpt_path)

        if getattr(self, "use_ema", False):
            self.model_ema = LitEma(self)
            print("[dual] EMA rebuilt over {} tensors".format(len(list(self.model_ema.buffers()))))

        # Single source of truth: the model's params drive the loss's routing.
        self.loss.dual_mode = self.dual_branch_mode
        self.loss.dual_split_level = self.dual_branch_split_level
        self.loss.dual_pad = self.dual_branch_pad
        self.loss.dual_steer_all_bands = self.dual_branch_steer_all_bands
        if self.dual_branch_steer_all_bands and self.dual_branch_mode != "replace":
            raise ValueError("dual_branch_steer_all_bands only applies to replace mode "
                             "(split already puts every configured band on the combined output)")
        n_new = sum(p.numel() for p in self.decoder_b2.parameters())
        print("[dual] mode={} split_level={} blocks={} pad={} -> branch 2 = {:.2f}M params".format(
            self.dual_branch_mode, self.dual_branch_split_level, self.dual_branch_blocks,
            self.dual_branch_pad, n_new / 1e6))

    def _load_branch2_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        pref = "decoder_b2."
        b2_sd = {k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}
        if not b2_sd:
            return                      # a plain (single-tail) checkpoint: the copy stands
        self.decoder_b2.load_state_dict(b2_sd, strict=True)
        print("[dual] branch 2 restored from {} ({} tensors)".format(path, len(b2_sd)))

    # ------------------------------------------------------- decoder, split in two
    def _decoder_trunk(self, z):
        """ldm...model.Decoder.forward (L535-561), stopping before the branch point."""
        d, temb = self.decoder, None
        h = d.conv_in(z)
        h = d.mid.block_1(h, temb)
        h = d.mid.attn_1(h)
        h = d.mid.block_2(h, temb)
        keep = d.num_res_blocks + 1 - self.dual_branch_blocks
        for i_level in reversed(range(d.num_resolutions)):
            n_here = (d.num_res_blocks + 1) if i_level != 0 else keep
            for i_block in range(n_here):
                h = d.up[i_level].block[i_block](h, temb)
                if len(d.up[i_level].attn) > 0:
                    h = d.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = d.up[i_level].upsample(h)
        return h

    def _tail(self, h, blocks, attn, norm_out, conv_out):
        temb = None
        for i, blk in enumerate(blocks):
            h = blk(h, temb)
            if len(attn) > 0:
                h = attn[i](h)
        h = norm_out(h)
        h = nonlinearity(h)
        h = conv_out(h)
        if self.decoder.tanh_out:
            h = torch.tanh(h)
        return h

    def decode_branches(self, z):
        z = self.post_quant_conv(z)
        h = self._decoder_trunk(z)
        keep = self.decoder.num_res_blocks + 1 - self.dual_branch_blocks
        b1 = self._tail(h, self.decoder.up[0].block[keep:], self.decoder.up[0].attn[keep:]
                        if len(self.decoder.up[0].attn) > 0 else [],
                        self.decoder.norm_out, self.decoder.conv_out)
        b2 = self._tail(h, self.decoder_b2.block, self.decoder_b2.attn,
                        self.decoder_b2.norm_out, self.decoder_b2.conv_out)
        out = band_split.mix(b1, b2, self.dual_branch_split_level, self.dual_branch_pad,
                             order=self.loss.steerable_order)
        return out, b1, b2

    def decode(self, z):
        out, b1, b2 = self.decode_branches(z)
        # Plain attribute on the loss, like teacher_input: never a buffer, never in
        # state_dict, overwritten every forward so it cannot go stale.
        self.loss.dual_branches = (b1, b2)
        return out

    def configure_optimizers(self):
        opts, scheds = super().configure_optimizers()
        opts[0].add_param_group({"params": list(self.decoder_b2.parameters())})
        n = sum(p.numel() for g in opts[0].param_groups for p in g["params"])
        print("[dual] opt_ae now covers {:.2f}M params (branch 2 included)".format(n / 1e6))
        return opts, scheds


class AutoencoderKLDualBranchEMANoGAN(DualBranchMixin, AutoencoderKLSteerableEMANoGAN):
    """The class these arms run: nerc-match recipe, no GAN, one optimiser."""


class AutoencoderKLDualBranchEMA(DualBranchMixin, AutoencoderKLSteerableEMA):
    """GAN variant, for a later arm. d_weight is measured at branch 1's conv_out
    (get_last_layer), which sees the adversarial gradient only through LP."""
