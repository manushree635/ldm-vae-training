from __future__ import annotations
"""L1 + LPIPS against the 512 teacher, in place of the steerable pyramid.

nmn1M distils the 512 teacher reconstruction through the steerable pyramid: the 256
student is bicubic-upsampled to 512 and compared band by band. This class keeps that
same target and the same upsample but compares in PIXELS and LPIPS instead -- so a
run with steerable_weight 0 and this term on differs from nmn1M only in the 512
representation (pyramid vs L1 + LPIPS).

    term = teacher512_l1_weight         * mean|up(S) - T|
         + teacher512_perceptual_weight * mean LPIPS(T, up(S))

`up` is the loss's own `_up`, i.e. the bicubic match_mode "upsample" the steerable
term uses (or the sub-pixel head if one is configured). Reductions are MEANS, the
same units as the 256 terms under nll_reduction: mean (logvar stays at its init 0
in the no-GAN class), so weight 1.0 here weighs the same as weight 1.0 at 256. The
LPIPS net is the loss's own perceptual_loss (perceptual_net picks VGG or AlexNet).

Runs on optimizer_idx 0 only, in train and val alike. Logged: <split>/t512_l1,
<split>/t512_lpips, <split>/t512_term, <split>/t512_share.
"""

import torch

from .losses_steerable import LPIPSWithDiscriminatorSteerable


class LPIPSWithDiscriminatorTeacher512(LPIPSWithDiscriminatorSteerable):
    def __init__(self, *args,
                 teacher512_l1_weight=1.0,
                 teacher512_perceptual_weight=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if self.nll_reduction != "mean":
            raise ValueError("teacher512 terms are mean-reduced; set nll_reduction: mean so the "
                             "256 terms are in the same units")
        self.teacher512_l1_weight = float(teacher512_l1_weight)
        self.teacher512_perceptual_weight = float(teacher512_perceptual_weight)
        print("[teacher512] L1 {} + LPIPS({}) {} vs the teacher at full res; steerable weight {}".format(
            self.teacher512_l1_weight, self.perceptual_net, self.teacher512_perceptual_weight,
            self.steerable_weight))

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split="train", weights=None):
        loss, log = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                                    global_step, last_layer=last_layer, cond=cond,
                                    split=split, weights=weights)
        if optimizer_idx != 0:
            return loss, log

        teacher = self.teacher_input
        if teacher is None:
            raise RuntimeError("teacher512 terms need a teacher: use AutoencoderKLSteerableEMA(NoGAN) "
                               "with a teacher dataset")
        if teacher.shape[0] != reconstructions.shape[0]:
            raise RuntimeError("teacher batch {} does not match reconstructions {}".format(
                tuple(teacher.shape), tuple(reconstructions.shape)))
        teacher = teacher.detach()
        up = self._up(reconstructions, teacher.shape[-2:])

        l1 = (up - teacher).abs().mean()
        if self.teacher512_perceptual_weight > 0:
            p = self.perceptual_loss(teacher.contiguous(), up.contiguous()).mean()
        else:
            p = torch.zeros_like(l1)
        term = self.teacher512_l1_weight * l1 + self.teacher512_perceptual_weight * p
        loss = loss + term

        log["{}/t512_l1".format(split)] = l1.detach()
        log["{}/t512_lpips".format(split)] = p.detach()
        log["{}/t512_term".format(split)] = term.detach()
        log["{}/t512_share".format(split)] = (term / loss).detach()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        return loss, log
