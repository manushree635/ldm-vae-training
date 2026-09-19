"""ft-MSE reconstruction loss for LDM's AutoencoderKL.

sd-vae-ft-ema and sd-vae-ft-mse differ in exactly one place: the pixel term of
LDM's LPIPSWithDiscriminator. Upstream hardcodes L1 (the ft-EMA recipe); ft-MSE
uses squared error with the perceptual weight dropped to 0.1.
"""

import torch

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
from taming.modules.losses.vqperceptual import adopt_weight


class LPIPSWithDiscriminatorMSE(LPIPSWithDiscriminator):
    """LDM's LPIPSWithDiscriminator with a selectable pixel reconstruction term.

    `forward` below is upstream's `contperceptual.LPIPSWithDiscriminator.forward`
    copied verbatim except for the single marked line, which upstream writes as

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

    The line is inlined at the top of a 60-line method rather than factored out,
    so overriding it means carrying the method. Everything else -- the logvar
    scaling, the KL term, the adaptive discriminator weight, the hinge loss, the
    log dict keys -- is upstream's, unchanged, so downstream tooling that reads
    `train/rec_loss` etc. keeps working.

    rec_loss_type="l1" reproduces upstream bit-for-bit.
    """

    def __init__(self, *args, rec_loss_type="mse", **kwargs):
        assert rec_loss_type in ("l1", "mse"), rec_loss_type
        super().__init__(*args, **kwargs)
        self.rec_loss_type = rec_loss_type

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        # >>> the only departure from upstream contperceptual.py <<<
        if self.rec_loss_type == "mse":
            rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2
        else:
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        # >>> end departure; the rest is upstream verbatim <<<
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log
