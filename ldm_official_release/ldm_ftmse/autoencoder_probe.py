from __future__ import annotations
"""Bottleneck probe: decode from the encoder's PRE-bottleneck features instead of the latent.

THE QUESTION. Every decoder-only arm converges on ~0.44 of the ground truth's fine detail being
correctly placed; the rest it invents. Is the missing detail still present one layer earlier, in
the encoder's 512-channel 32x32 features, and merely discarded by the gate `encoder.conv_out`
(512 -> 8, which throws away 524,288 numbers per image and keeps 4,096)? Or is it already gone
further up the encoder, in which case no bottleneck change can recover it?

THE TEST. Train the ordinary decoder, from its pretrained weights, to reconstruct from those
512-channel features instead of from z. Score it with the same invented-detail metric as every
other arm and compare against the matched decoder-only control (`mctrl`), which is the identical
recipe decoding from z.

  probe >> control  ->  the information survives to the gate; retraining the gate is well posed.
  probe ~= control  ->  the encoder trunk already lost it; changing the bottleneck cannot help.

THE ENCODER STAYS FROZEN. This measures the frozen encoder's features; it does not train a new
encoder. Only the decoder and one new 512->512 injection conv train. That conv is 2.36M params
against the control's 18k (post_quant_conv + conv_in), so the probe carries slightly more
capacity -- a caveat, though the consistency-decoder result (13x capacity, no gain in real
detail) says capacity is not what moves this number.
"""

import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.model import nonlinearity
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma

from .autoencoder_steerable import AutoencoderKLSteerableEMA


class AutoencoderKLBottleneckProbe(AutoencoderKLSteerableEMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        block_in = self.encoder.conv_out.in_channels          # 512 for kl-f8
        # Replaces decoder.conv_in (4 -> 512) on the probe path. Built AFTER super().__init__,
        # so init_from_ckpt has already run and does not see it as a missing key.
        self.probe_in = nn.Conv2d(block_in, block_in, kernel_size=3, stride=1, padding=1)
        self.probe_in.requires_grad_(True)
        # LitEma only shadows params that had requires_grad when IT was built, and the parent
        # built it before probe_in existed. Rebuild, or the EMA weights used at validation and
        # export would carry a random injection conv.
        if getattr(self, "use_ema", False):
            self.model_ema = LitEma(self)
            print(f"[probe] EMA rebuilt over {len(list(self.model_ema.buffers()))} tensors")
        print(f"[probe] decoding from encoder pre-bottleneck features "
              f"({block_in}x32x32), not z; probe_in = "
              f"{sum(p.numel() for p in self.probe_in.parameters())/1e6:.2f}M params")

    # --- ldm.modules.diffusionmodules.model.Encoder.forward, minus conv_out ---------------
    def _encoder_trunk(self, x):
        e, temb = self.encoder, None
        hs = [e.conv_in(x)]
        for i_level in range(e.num_resolutions):
            for i_block in range(e.num_res_blocks):
                h = e.down[i_level].block[i_block](hs[-1], temb)
                if len(e.down[i_level].attn) > 0:
                    h = e.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != e.num_resolutions - 1:
                hs.append(e.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = e.mid.block_1(h, temb)
        h = e.mid.attn_1(h)
        h = e.mid.block_2(h, temb)
        h = e.norm_out(h)
        return nonlinearity(h)                 # exactly what conv_out is handed

    # --- ldm...model.Decoder.forward, with conv_in replaced by probe_in -------------------
    def _decoder_tail(self, feat):
        d, temb = self.decoder, None
        h = self.probe_in(feat)
        h = d.mid.block_1(h, temb)
        h = d.mid.attn_1(h)
        h = d.mid.block_2(h, temb)
        for i_level in reversed(range(d.num_resolutions)):
            for i_block in range(d.num_res_blocks + 1):
                h = d.up[i_level].block[i_block](h, temb)
                if len(d.up[i_level].attn) > 0:
                    h = d.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = d.up[i_level].upsample(h)
        if d.give_pre_end:
            return h
        h = d.norm_out(h)
        h = nonlinearity(h)
        h = d.conv_out(h)
        if d.tanh_out:
            h = torch.tanh(h)
        return h

    def _probe(self, x):
        with torch.no_grad():                                  # encoder is frozen
            feat = self._encoder_trunk(x)
            posterior = DiagonalGaussianDistribution(
                self.quant_conv(self.encoder.conv_out(feat)))   # kept for the KL term / logging
        posterior.sample()                                      # keep the RNG draw the control makes
        return self._decoder_tail(feat), posterior

    def forward(self, input, sample_posterior=True):
        return self._probe(input)

    def _ae_forward(self, batch, inputs):
        # NB: deliberately ignores the latent cache -- the probe needs the features, not z.
        return self._probe(inputs)

    def configure_optimizers(self):
        """Parent lists the trainable modules explicitly, so probe_in must be added by hand."""
        opts, scheds = super().configure_optimizers()
        opts[0].add_param_group({"params": list(self.probe_in.parameters())})
        n = sum(p.numel() for g in opts[0].param_groups for p in g["params"])
        print(f"[probe] opt_ae now covers {n/1e6:.2f}M params (probe_in included)")
        return opts, scheds
