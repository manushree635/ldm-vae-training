from __future__ import annotations
"""Decoder-only + EMA + GAN autoencoder that also feeds a teacher to the loss."""

import torch

from .autoencoder import AutoencoderKLDecoderOnlyEMA


class AutoencoderKLSteerableEMA(AutoencoderKLDecoderOnlyEMA):
    """AutoencoderKLDecoderOnlyEMA that stashes a teacher tensor for the loss.

    There are FIVE self.loss(...) call sites: training_step opt-0 and opt-1,
    validation_step 0 and 1, and AutoencoderKLDecoderOnlyEMA.validation_step's EMA
    pass. The EMA one routes through super().validation_step, so threading an
    explicit `teacher=` kwarg would mean carrying two whole method bodies -- and the
    EMA site is precisely the one that gets forgotten.

    Stashing on the loss module covers all five at once. `teacher_input` is a plain
    attribute on the loss (nn.Module only diverts a Tensor into _buffers if the name
    is already registered), so it never reaches state_dict(), LitEma's shadow, or
    init_from_ckpt's missing-key assertion. It is overwritten at the top of every
    step, so it cannot go stale; it is deliberately NOT popped, because
    validation_step legitimately consumes it three times.

    This class adds no Modules and no Parameters, so the checkpoint and EMA key sets
    stay byte-identical to the plain EMA+GAN run.
    """

    def __init__(self, *args, teacher_key="teacher", **kwargs):
        # Set before super(): __init__ builds the loss and may touch attributes.
        self.teacher_key = teacher_key
        super().__init__(*args, **kwargs)

    def _stash_teacher(self, batch):
        if self.teacher_key in batch:
            self.loss.teacher_input = self.get_input(batch, self.teacher_key).detach()
        # The band schedule is keyed on epoch; the loss has no other way to know it.
        # Plain int assignment, so it never becomes a buffer or enters state_dict.
        if hasattr(self.loss, "current_epoch"):
            self.loss.current_epoch = int(self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self._stash_teacher(batch)
        return super().training_step(batch, batch_idx, optimizer_idx)

    def validation_step(self, batch, batch_idx):
        self._stash_teacher(batch)
        return super().validation_step(batch, batch_idx)

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = super().log_images(batch, only_inputs=only_inputs, **kwargs)
        if self.teacher_key in batch:
            # Free ongoing alignment check: the teacher grid should visibly match the
            # inputs grid image-for-image in logs/<run>/images/train/.
            log["teacher"] = self.get_input(batch, self.teacher_key).to(self.device)
        return log
