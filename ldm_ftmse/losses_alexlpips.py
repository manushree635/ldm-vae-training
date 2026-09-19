from __future__ import annotations
"""Upstream's LPIPSWithDiscriminator with the LPIPS trunk swapped to AlexNet.

Upstream hardcodes `self.perceptual_loss = LPIPS().eval()` (VGG16) at
contperceptual.py:17 with no switch. Subclassing and reassigning that one
attribute in __init__ is the whole change: every other line of the parent --
the l1 rec_loss, the logvar division, the sum/batch reduction, kl_weight,
calculate_adaptive_weight, adopt_weight/disc_factor gating, both optimizer
branches -- runs untouched.

Reassignment is safe here because the parent stores LPIPS as a plain submodule
attribute and only ever calls it as `self.perceptual_loss(inputs, recons)`.
LPIPSAlex matches that contract exactly: same [-1,1] input convention, same
[N,1,1,1] output. Verified bit-identical to richzhang/PerceptualSimilarity's
lpips.LPIPS(net='alex') -- max abs diff 0.0 on a 4x3x256x256 probe.

NOTE the perceptual scale differs between trunks. On the same probe alex returns
~0.199 where vgg returns a larger value, so `perceptual_weight` is NOT directly
transferable between them: a run swapping the trunk at fixed perceptual_weight
is also changing the effective perceptual:l1 balance. That is a real confound to
keep in mind when reading the result, not something this class can fix.
"""

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator

from .lpips_alex import LPIPSAlex


class LPIPSAlexWithDiscriminator(LPIPSWithDiscriminator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perceptual_loss = LPIPSAlex().eval()
        for p in self.perceptual_loss.parameters():
            p.requires_grad = False
        print("[alex-lpips] perceptual trunk = AlexNet (was VGG16)")
