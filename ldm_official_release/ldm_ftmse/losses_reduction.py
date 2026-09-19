from __future__ import annotations
"""LPIPSWithDiscriminator with the reduction (sum-per-image vs mean) exposed.

Upstream hardcodes a per-image SUM at contperceptual.py:57-61:

    weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
    nll_loss          = torch.sum(nll_loss)          / nll_loss.shape[0]
    kl_loss           = torch.sum(kl_loss)           / kl_loss.shape[0]

so nll is summed over 3*256*256 = 196,608 elements per image and kl over
4*32*32 = 4,096. Most reimplementations (SpectrumMatching among them) use a plain
`.mean()` instead. This class makes that a config switch rather than a fork.

WHAT ACTUALLY CHANGES, given this project's settings -- checked, not assumed:

  logvar  INERT. logvar_init=0 and loss.logvar is in NEITHER optimizer (only
          decoder + post_quant_conv, and the discriminator), so exp(logvar)=1 and
          the "+ logvar" term is 0 in both modes. Confirmed: train/logvar is
          0 at first and last step of the control run. Under mean reduction the
          "+ logvar" term would otherwise have shifted from 196,608*logvar to
          1*logvar per image, so this being dead removes a real confound.

  d_weight SELF-COMPENSATING. d_weight = ||grad nll|| / (||grad g|| + 1e-4).
          Scaling nll by 1/196608 scales its gradient by the same factor, so
          d_weight scales down identically and `d_weight * g_loss` keeps its
          ratio to nll. The clamp at 1e4 would break that if it bound -- but
          measured d_weight on the control runs 3780 -> 1569, comfortably under
          the 5000 ceiling (1e4 * disc_weight 0.5), so it does not.

  kl      RATIO SHIFTS 196,608x -- AND THIS ONE BITES. DiagonalGaussianDistribution
          .kl() already sums over dim=[1,2,3] and returns shape [N], one value per
          image. For a per-image tensor `sum(x)/x.shape[0]` IS `mean(x)`, so the
          reduction leaves kl COMPLETELY UNCHANGED while nll shrinks by 196,608.
          Verified: kl_loss is 3389.17 in both modes on the same input.
          Consequence at kl_weight=1e-6, measured on the control run:
              sum : 1e-6 * 3.4e3 = 3.4e-3  against nll 2.1e4  -> 0.00002% of loss
              mean: 1e-6 * 3.4e3 = 3.4e-3  against nll 0.105   -> 3.2%     of loss
          So plain mean reduction quietly promotes KL from irrelevant to a few
          percent of the objective. Set kl_weight_compensate=True to divide
          kl_weight by the image element count and hold the balance fixed.

SO THE REAL QUESTION IS THE OPTIMISER, NOT THE OBJECTIVE. With the above, mean
reduction is the same objective scaled by 1/196608. Adam is scale-invariant in
principle -- m/sqrt(v) is unchanged by a global gradient scale -- EXCEPT for its
epsilon. Default eps=1e-8; if sum-mode gradients sit around 1e-3, mean-mode ones
land near 5e-9, i.e. BELOW eps, at which point eps dominates the denominator and
silently shrinks every update. That is the thing this switch is worth testing,
and it is why `adam_eps` is plumbed through the model, not fixed here.
"""

import torch

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator, adopt_weight


class LPIPSWithDiscriminatorReduction(LPIPSWithDiscriminator):
    """Upstream's loss with `reduction` switchable between 'sum' and 'mean'.

    Args:
        reduction: 'sum' reproduces upstream exactly (per-image sum over all
            elements); 'mean' divides additionally by the element count.
        kl_weight_compensate: with reduction='mean', DIVIDE kl_weight by the
            image element count (3*256*256 = 196,608) so the nll:kl balance
            matches sum mode. Off by default, which is faithful to what other
            implementations do -- but see the kl note above: leaving it off is
            not a no-op, it makes KL ~3% of the objective instead of ~0%.
    """

    def __init__(self, *args, reduction="sum", kl_weight_compensate=False, **kwargs):
        super().__init__(*args, **kwargs)
        if reduction not in ("sum", "mean"):
            raise ValueError("reduction must be 'sum' or 'mean', got {!r}".format(reduction))
        self.reduction = reduction
        self.kl_weight_compensate = bool(kl_weight_compensate)
        self._kl_comp = 1.0          # set on first forward, once shapes are known
        print("[reduction] nll/kl reduction = {}{}".format(
            reduction, "  (+kl_weight compensation)" if kl_weight_compensate else ""))

    def _reduce(self, x):
        """Upstream's per-image sum, or a full mean."""
        if self.reduction == "sum":
            return torch.sum(x) / x.shape[0]
        return torch.mean(x)

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", weights=None):
        # --- identical to upstream through rec_loss -------------------------
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss

        # --- the only change: reduction ------------------------------------
        weighted_nll_loss = self._reduce(weighted_nll_loss)
        nll_loss = self._reduce(nll_loss)
        kl_loss = self._reduce(posteriors.kl())

        kl_weight = self.kl_weight
        if self.reduction == "mean" and self.kl_weight_compensate:
            if self._kl_comp == 1.0:
                # nll shrank by this factor; kl did not (kl() is already per-image),
                # so cancel the shift by scaling kl_weight the same way.
                n_img = inputs[0].numel()
                self._kl_comp = 1.0 / float(n_img)
                print("[reduction] kl_weight compensation /{} "
                      "(image elems) -> {:.3e}".format(n_img, self.kl_weight * self._kl_comp))
            kl_weight = kl_weight * self._kl_comp

        # --- everything below is upstream, verbatim -------------------------
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step,
                                       threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step,
                                       threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}
            return d_loss, log
