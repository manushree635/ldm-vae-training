from __future__ import annotations
"""Steerable AutoencoderKL with the discriminator held FIXED.

Not the same as switching the GAN off. In upstream's loss a single `disc_factor`
gates both branches --

    generator:      loss  = nll + kl*kl_weight + d_weight * disc_factor * g_loss
    discriminator:  d_loss =                                 disc_factor * disc_loss

-- so disc_factor=0 removes the adversarial signal from the decoder as well.
This class keeps that signal and only stops the critic from moving, giving a
constant adversarial objective instead of a co-adapting one.

HOW IT FREEZES, AND WHY NOT THE OBVIOUS WAY.

The first attempt did the obvious thing -- requires_grad_(False) on the
discriminator plus `return None` from training_step for optimizer_idx == 1 --
and it dies under DDP:

    RuntimeError: Expected to mark a variable ready only once.
    Parameter at index 138 has been marked as ready twice.

With two optimisers DDP expects a matched forward/backward for each optimiser
index every iteration; skipping one leaves the reducer's accounting inconsistent
and the next backward re-marks parameters already marked. It runs fine on a
single GPU, so the failure only appears at 2+ ranks.

So instead the discriminator branch runs exactly as upstream intends -- forward,
backward, optimiser step -- and the step is made a no-op by giving opt_disc a
learning rate of ZERO. Adam's update is -lr * m/(sqrt(v)+eps), so lr=0 leaves
every weight bit-identical (torch.optim.Adam defaults to weight_decay=0, so
there is no decay path either). DDP sees the pattern it expects.

Two consequences worth knowing:

  * this does NOT save time. The earlier version skipped the second autoencoder
    forward as a side effect, which was worth ~25% of step time; that saving is
    forfeited to keep DDP happy.
  * requires_grad stays True, so the EMA shadow still covers the discriminator.
    Harmless -- it averages constants -- and it avoids the LitEma ordering trap
    that killed the previous attempt (LitEma shadows what has requires_grad at
    construction, and copy_to asserts that frozen params are absent from it).

BatchNorm still needs handling separately: NLayerDiscriminator's running_mean /
running_var update on every FORWARD pass regardless of gradients or learning
rate. Measured on the control arm, num_batches_tracked reached 1188 after one
epoch while the learned weights were still bit-identical. Holding the module in
eval() -- and re-asserting it after every Lightning .train() call -- keeps the
statistics pinned too, so "frozen" means the critic is a constant function.
"""

import torch

from .autoencoder_steerable import AutoencoderKLSteerableEMA


class AutoencoderKLSteerableEMAFrozenDisc(AutoencoderKLSteerableEMA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss.discriminator.eval()
        n = sum(p.numel() for p in self.loss.discriminator.parameters())
        print(f"[frozen-disc] discriminator pinned: {n/1e6:.2f}M parameters, "
              f"eval mode, opt_disc lr=0")

    def train(self, mode=True):
        super().train(mode)
        self.loss.discriminator.eval()      # keep BN statistics frozen too
        return self

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.decoder.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        # lr=0 -> Adam's update is identically zero, so the critic never moves,
        # while the branch still runs and DDP sees a matched forward/backward.
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=0.0, betas=(0.5, 0.9))
        print(f"[frozen-disc] opt_ae lr={lr}  opt_disc lr=0.0 (critic frozen)")
        return [opt_ae, opt_disc], []
