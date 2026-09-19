from __future__ import annotations
"""Decoder-only + EMA + steerable autoencoder with the GAN removed entirely.

Upstream's AutoencoderKL always builds two optimizers (autoencoder + critic) and
runs every batch twice, once per optimizer index. Setting disc_factor=0 zeroes the
adversarial terms but keeps all of that machinery -- the critic still runs forward,
the batch still goes through the model twice, and Adam still carries GAN-tuned
betas (0.5, 0.9).

This class is the no-GAN recipe distillation_nerc uses by default:

  * ONE optimizer over decoder + post_quant_conv (+ the sub-pixel head if any).
    PL 1.4.2 only passes `optimizer_idx` when there is more than one optimizer
    (training_batch_loop.py:650), so training_step runs once per batch.
  * Adam with torch's DEFAULT betas (0.9, 0.999) unless `adam_betas` is given.
    (0.5, 0.9) is a GAN choice -- a ~10-step second-moment window so the optimiser
    keeps up with a moving critic. Without a critic there is no reason for it;
    distillation_nerc's no-GAN path uses the defaults too (lightning_module.py:2850).
  * The critic is frozen and held in eval(): its weights come in with the
    checkpoint but never enter any graph or optimiser.

LitEma ORDERING. The parent builds LitEma in __init__, while the discriminator still
has requires_grad=True, so its shadow would include the critic -- and
LitEma.copy_to asserts that frozen params are ABSENT from the shadow (this exact
trap killed the first frozen-disc attempt). So the EMA is rebuilt here, after the
freeze, the same way AutoencoderKLDecoderOnlyEMA builds it after the encoder
freeze.

Requires lossconfig.params.disc_factor: 0.0 -- refused otherwise, because a
non-zero disc_factor with no critic optimiser would be a silent half-GAN.
"""

import torch
from ldm.modules.ema import LitEma

from .autoencoder_steerable import AutoencoderKLSteerableEMA


class AutoencoderKLSteerableEMANoGAN(AutoencoderKLSteerableEMA):
    def __init__(self, *args, adam_betas=None, adam_weight_decay=0.0, grad_clip_norm=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        # grad_clip_norm: global L2 clip on the TRUE (unscaled) gradients -- see
        # on_before_optimizer_step. nerc: training_config.grad_clip_norm = 1.0.
        self.grad_clip_norm = float(grad_clip_norm)
        self._last_grad_norm = None
        self.adam_betas = tuple(float(b) for b in adam_betas) if adam_betas is not None else None
        # nerc passes weight_decay=1e-5 to its no-GAN Adam (lightning_module.py:2850).
        self.adam_weight_decay = float(adam_weight_decay)

        if float(getattr(self.loss, "disc_factor", 0.0)) != 0.0:
            raise ValueError("AutoencoderKLSteerableEMANoGAN needs lossconfig disc_factor: 0.0 "
                             "(got {})".format(self.loss.disc_factor))

        disc = self.loss.discriminator
        disc.requires_grad_(False)
        disc.eval()
        n_disc = sum(p.numel() for p in disc.parameters())

        if self.use_ema:
            self.model_ema = LitEma(self)          # rebuilt AFTER the freeze -- see docstring
            print(f"[no-GAN] critic frozen ({n_disc/1e6:.2f}M params), "
                  f"EMA rebuilt: {len(list(self.model_ema.buffers()))} buffers")

    def on_train_start(self):
        # PL 1.4.2 + precision=16 clips BEFORE unscaling: native_amp.pre_optimizer_step
        # runs the closure (backward -> _track_and_norm_grad -> clip_gradients) and only
        # then calls scaler.unscale_. Trainer(gradient_clip_val=c) would therefore clip at
        # c / loss_scale (65536 at start) -- effectively freezing the model. Refuse it;
        # use grad_clip_norm on this module instead.
        t = self.trainer
        if float(getattr(t, "gradient_clip_val", 0) or 0) > 0 and int(getattr(t, "precision", 32)) == 16:
            raise RuntimeError("Trainer gradient_clip_val with precision=16 clips SCALED grads on "
                               "PL 1.4.2; set model.params.grad_clip_norm instead")

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        # Called from PrecisionPlugin.pre_optimizer_step, which the native-AMP plugin
        # invokes AFTER scaler.unscale_(optimizer) -- so these are the true gradients,
        # already all-reduced by DDP and summed over accumulation. Same point nerc
        # (Lightning 2) clips at. An overflow step has inf grads; scaler.step skips it.
        if self.grad_clip_norm <= 0:
            return
        params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
        if params:
            self._last_grad_norm = torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm).detach()
            if self.global_step < 3:
                print(f"[no-GAN] step {self.global_step}: grad norm pre-clip "
                      f"{float(self._last_grad_norm):.4g} (clip {self.grad_clip_norm})")

    def train(self, mode=True):
        super().train(mode)
        self.loss.discriminator.eval()             # keep its BatchNorm stats frozen too
        return self

    def configure_optimizers(self):
        lr = self.learning_rate
        groups = [{"params": list(self.decoder.parameters()) +
                             list(self.post_quant_conv.parameters())}]
        up_head = getattr(self.loss, "up_head", None)
        if up_head is not None:
            mult = float(getattr(self.loss, "steerable_upsampler_lr_mult", 1.0))
            groups.append({"params": list(up_head.parameters()), "lr": lr * mult})
        kw = {} if self.adam_betas is None else {"betas": self.adam_betas}
        opt = torch.optim.Adam(groups, lr=lr, weight_decay=self.adam_weight_decay, **kw)
        print(f"[no-GAN] single Adam optimiser, lr={lr}, betas={opt.defaults['betas']}, "
              f"weight_decay={opt.defaults['weight_decay']}")
        return [opt], []

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self._stash_teacher(batch)
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self._ae_forward(batch, inputs)
        aeloss, log_dict = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                     last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self._last_grad_norm is not None:            # from the previous optimiser step
            self.log("train/grad_norm_pre_clip", self._last_grad_norm, logger=True,
                     on_step=True, on_epoch=False)
        self.log_dict(log_dict, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss
