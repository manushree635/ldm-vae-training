from __future__ import annotations
"""Whole-VAE finetune: the ENCODER trains too, held near its starting latent by a scale-free anchor.

Everything else in ldm_ftmse is decoder-only (autoencoder.py freezes encoder + quant_conv, matching
sd-vae-ft-mse). The gate probes (distillation_nerc/analysis/stage_ledger/gate_probe.py, 2026-09-17)
showed that the fine bands a 4-channel latent loses are lost in the ENCODER TRUNK, and that only
retraining the trunk recovers them: L1 corr^2 0.517 -> 0.658 (whole encoder) vs 0.549 for a trunk-sized
gate after Emid. A light anchor on z kept most of that (0.626) at a fifth of the latent drift.

This class is AutoencoderKLSteerableEMA (teacher stash, EMA, GAN, steerable loss -- all unchanged) with
four differences, and adds NO parameters, so the state_dict key set is identical to the decoder-only arms
and scripts/export_ldm_vae_to_diffusers.py works as-is (it converts every tensor, encoder included):

  1. encoder + quant_conv are UNFROZEN after the checkpoint loads, and train in opt_ae
     (their own param group, lr * encoder_lr_mult).
  2. LitEma is REBUILT after the unfreeze. The parent builds it while the encoder is frozen, and LitEma
     only shadows requires_grad params -- without the rebuild the EMA (and --use-ema exports) would
     silently keep the ORIGINAL encoder next to an EMA decoder trained for a different one.
  3. A frozen copy of the INITIAL encoder + quant_conv is the anchor reference. It is held in a plain
     list, NOT registered, so it never reaches state_dict, LitEma, the optimiser, or the exporter.
  4. anchor loss, added to the generator loss only:
         anchor_mode "std" (default): mean over channels of MSE between the per-channel batch-
             STANDARDISED z and z_ref = 2(1 - corr). Scale and offset are free (latent rescaling absorbs
             them); only restructuring is penalised. This is gate_probe.py's `--anchor_mode std`.
         anchor_mode "raw": relative MSE (z - z_ref)^2 / var(z_ref).
     z is the posterior MEAN (the posterior sd is ~4e-4, so mean vs sample is immaterial).
     anchor_normalize "nll_units" (default) multiplies by C*H*W of the input, the same per-image-SUM units
     the steerable term uses (losses_steerable.py, `steerable_normalize: nll_units`), so
     anchor_weight means what it meant in the probe: 0.1 ~ a few % of the objective at the probe's
     equilibrium drift of ~0.06.

GUARD: the latent cache (data `latents_file`, batch key `moments`) holds the FROZEN encoder's output; with a
training encoder it is stale on every step. _ae_forward refuses it rather than train on it silently.

train() keeps the encoder in train mode (the parent forces encoder.eval(); dropout is 0.0 in kl-f8, so this
changes no numbers, but a trainable module should not be pinned to eval).
"""

import copy

import torch

from ldm.modules.ema import LitEma

from .autoencoder import LATENTS_KEY, AutoencoderKLDecoderOnly
from .autoencoder_steerable import AutoencoderKLSteerableEMA

_ANCHOR_MODES = ("std", "raw")
_ANCHOR_NORMS = ("nll_units", "none")


class AutoencoderKLFullEncAnchorEMA(AutoencoderKLSteerableEMA):

    def __init__(self, *args, anchor_weight=0.1, anchor_mode="std", anchor_normalize="nll_units",
                 anchor_eps=1e-6, encoder_lr_mult=1.0, **kwargs):
        super().__init__(*args, **kwargs)      # loads ckpt_path, freezes the encoder, builds the (decoder) EMA
        if anchor_mode not in _ANCHOR_MODES:
            raise ValueError(f"anchor_mode must be one of {_ANCHOR_MODES}, got {anchor_mode!r}")
        if anchor_normalize not in _ANCHOR_NORMS:
            raise ValueError(f"anchor_normalize must be one of {_ANCHOR_NORMS}, got {anchor_normalize!r}")
        self.anchor_weight = float(anchor_weight)
        self.anchor_mode = anchor_mode
        self.anchor_normalize = anchor_normalize
        self.anchor_eps = float(anchor_eps)
        self.encoder_lr_mult = float(encoder_lr_mult)

        # (3) the anchor reference: the encoder AS LOADED, frozen, unregistered.
        ref = torch.nn.ModuleDict({"encoder": copy.deepcopy(self.encoder),
                                   "quant_conv": copy.deepcopy(self.quant_conv)})
        ref.requires_grad_(False)
        ref.eval()
        self._anchor_ref = [ref]               # a list, so nn.Module does not register it

        # (1) unfreeze
        self.encoder.requires_grad_(True)
        self.quant_conv.requires_grad_(True)

        # (2) rebuild the EMA so it shadows the encoder too
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"[fullenc] EMA rebuilt after unfreezing the encoder: "
                  f"{len(list(self.model_ema.buffers()))} buffers")
        n_enc = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.quant_conv.parameters())
        print(f"[fullenc] encoder + quant_conv TRAINABLE ({n_enc / 1e6:.2f}M, lr x{self.encoder_lr_mult}); "
              f"anchor {self.anchor_mode} w={self.anchor_weight} ({self.anchor_normalize})")

    # ---------------------------------------------------------------- mode / optimiser
    def train(self, mode=True):
        # skip AutoencoderKLDecoderOnly.train's encoder.eval(): the encoder trains here
        super(AutoencoderKLDecoderOnly, self).train(mode)
        return self

    def configure_optimizers(self):
        lr = self.learning_rate
        groups = [{"params": list(self.decoder.parameters()) + list(self.post_quant_conv.parameters())},
                  {"params": list(self.encoder.parameters()) + list(self.quant_conv.parameters()),
                   "lr": lr * self.encoder_lr_mult}]
        up_head = getattr(self.loss, "up_head", None)
        if up_head is not None:
            mult = float(getattr(self.loss, "steerable_upsampler_lr_mult", 1.0))
            groups.append({"params": list(up_head.parameters()), "lr": lr * mult})
        opt_ae = torch.optim.Adam(groups, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
        n_opt = sum(p.numel() for g in groups for p in g["params"])
        print(f"[fullenc] opt_ae optimises {n_opt / 1e6:.2f}M params (decoder + post_quant_conv + "
              f"encoder + quant_conv{' + up_head' if up_head is not None else ''}) at lr={lr}")
        return [opt_ae, opt_disc], []

    # ---------------------------------------------------------------- guard
    def _ae_forward(self, batch, inputs):
        if isinstance(batch, dict) and batch.get(LATENTS_KEY) is not None:
            raise RuntimeError(
                "AutoencoderKLFullEncAnchorEMA: the batch carries cached encoder moments "
                f"('{LATENTS_KEY}'), but the encoder is TRAINING, so the cache is stale. "
                "Remove `latents_file` from the data config.")
        return self(inputs)

    # ---------------------------------------------------------------- anchor
    def _ref(self, device):
        ref = self._anchor_ref[0]
        if next(ref.parameters()).device != device:
            ref.to(device)
        return ref

    def anchor_drift(self, inputs, posterior):
        """(drift, weighted loss). drift is 2(1-corr) ('std') or relative MSE ('raw') vs the initial z."""
        ref = self._ref(inputs.device)
        with torch.no_grad():
            m = ref["quant_conv"](ref["encoder"](inputs))
            zr = m[:, :m.shape[1] // 2].float()
        z = posterior.mean.float()
        if self.anchor_mode == "std":
            def st(t):
                return (t - t.mean(dim=(0, 2, 3), keepdim=True)) / (t.std(dim=(0, 2, 3), keepdim=True) + self.anchor_eps)
            drift = ((st(z) - st(zr)) ** 2).mean()
        else:
            drift = (((z - zr) ** 2).mean(dim=(0, 2, 3)) / (zr.var(dim=(0, 2, 3)) + self.anchor_eps)).mean()
        scale = float(inputs[0].numel()) if self.anchor_normalize == "nll_units" else 1.0
        return drift, self.anchor_weight * scale * drift

    # ---------------------------------------------------------------- steps
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx != 0:
            return super().training_step(batch, batch_idx, optimizer_idx)   # discriminator: unchanged
        self._stash_teacher(batch)
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self._ae_forward(batch, inputs)
        if self.disc_reuse_recon:
            self._recon_cache = reconstructions.detach()
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx,
                                        self.global_step, last_layer=self.get_last_layer(),
                                        split="train")
        drift, anchor = self.anchor_drift(inputs, posterior)
        total = aeloss + anchor if self.anchor_weight > 0 else aeloss
        self.log("aeloss", total, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.log_dict({"train/anchor_drift": drift.detach(), "train/anchor_loss": anchor.detach(),
                       "train/anchor_share": (anchor.detach() / total.detach().abs().clamp_min(1e-12))},
                      prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return total

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        inputs = self.get_input(batch, self.image_key)
        posterior = self.encode(inputs)
        drift, _ = self.anchor_drift(inputs, posterior)
        self.log("val/anchor_drift", drift)
        return out
