from __future__ import annotations
"""Steerable distillation on top of an MSE reconstruction loss instead of L1.

Both pieces already exist and compose without reimplementing either:

  LPIPSWithDiscriminatorMSE (ldm_ftmse/losses.py) reimplements upstream's forward
  with one departure -- rec_loss is (x-y)**2 rather than |x-y| when
  rec_loss_type="mse". At rec_loss_type="l1" it is bit-identical to upstream
  (verified delta 0.00e+00 on both optimizer branches).

  LPIPSWithDiscriminatorSteerable (ldm_ftmse/losses_steerable.py) does NOT
  reimplement the base loss. Its forward calls `super().forward(...)` for the
  L1/LPIPS/KL/GAN part and then adds the steerable term to whatever came back.

So the combination is just an MRO question. With

    class ...SteerableMSE(Steerable, MSE)

the resolution order is SteerableMSE -> Steerable -> MSE -> LPIPSWithDiscriminator.
`Steerable.forward` runs, its `super().forward(...)` lands on `MSE.forward`
(which supplies the squared-error rec_loss), and the steerable term is added on
top. `Steerable.__init__` passes unknown kwargs through, so `rec_loss_type`
reaches `MSE.__init__` from the config unchanged.

WHY IT MATTERS HERE, beyond swapping a norm: upstream sums rec_loss per image
over 3*256*256 elements. |x-y| and (x-y)^2 are on different scales -- for
residuals below 1 the square is SMALLER -- so the nll term shrinks relative to
the steerable term, which is normalised separately by `nll_units`. The same
steerable_weight therefore buys a LARGER share of the objective under MSE. Check
train/steerable_share against the L1 arms before attributing any difference to
the norm itself.
"""

from .losses import LPIPSWithDiscriminatorMSE
from .losses_steerable import LPIPSWithDiscriminatorSteerable


class LPIPSWithDiscriminatorSteerableMSE(LPIPSWithDiscriminatorSteerable,
                                         LPIPSWithDiscriminatorMSE):
    """Steerable term + MSE reconstruction loss. Set rec_loss_type in the config."""
