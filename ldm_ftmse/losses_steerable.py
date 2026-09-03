from __future__ import annotations
"""Upstream's LPIPSWithDiscriminator plus a steerable-pyramid distillation term.

Only the steerable term uses the teacher. L1, LPIPS and the discriminator keep
using the ground-truth 256px image, which is what distillation_nerc does too --
`use_gt_for_image_losses: false` there never affected LPIPS/L1, and the
discriminator's "real" is always the GT image (lightning_module.py:2529). Feeding
the teacher to the discriminator would train the student to look like a blurry
reconstruction rather than a real photograph.

UNITS. Upstream reduces with `torch.sum(nll)/N` -- a per-image SUM over
3*256*256 = 196,608 elements -- so total_loss sits around 29,000 in the live run.
The steerable term is a MEAN, around 0.07. Added raw with weight 0.5 it would be
~1.2e-6 of the objective, i.e. numerically absent. `steerable_normalize:
nll_units` multiplies by the element count so the weight means what it meant in
distillation_nerc, where every loss is mean-reduced.
"""

import torch

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator

from .band_schedule import active_levels_at, resolve_band_weights
from .steerable_rel_l1 import compute_steerable_rel_l1


class LPIPSWithDiscriminatorSteerable(LPIPSWithDiscriminator):
    def __init__(self, *args,
                 steerable_weight=0.5,
                 steerable_normalize="nll_units",   # "nll_units" | "none"
                 steerable_height=5,
                 steerable_order=3,
                 steerable_band_weights=None,
                 steerable_match_mode="upsample",
                 steerable_color_mode="luminance",
                 steerable_complex=True,
                 steerable_complex_metric="magnitude",
                 steerable_eps=1e-8,
                 steerable_start=0,
                 steerable_log_bands=False,
                 steerable_band_schedule=None,
                 steerable_band_ramp_epochs=0,
                 steerable_band_schedule_order="coarse_to_fine",
                 steerable_normalize_by_active_levels=False,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # OmegaConf hands us a DictConfig, which is NOT a dict -- _resolve_band_weights
        # would hit its TypeError. Convert once, here.
        if steerable_band_weights is not None and not isinstance(steerable_band_weights, (dict, list, tuple)):
            from omegaconf import OmegaConf
            steerable_band_weights = OmegaConf.to_container(steerable_band_weights, resolve=True)

        if steerable_normalize not in ("nll_units", "none"):
            raise ValueError("steerable_normalize must be 'nll_units' or 'none'")

        self.steerable_weight = float(steerable_weight)
        self.steerable_normalize = steerable_normalize
        self.steerable_height = int(steerable_height)
        self.steerable_order = int(steerable_order)
        self.steerable_band_weights = steerable_band_weights
        self.steerable_match_mode = steerable_match_mode
        self.steerable_color_mode = steerable_color_mode
        self.steerable_complex = bool(steerable_complex)
        self.steerable_complex_metric = steerable_complex_metric
        self.steerable_eps = float(steerable_eps)
        self.steerable_start = int(steerable_start)
        self.steerable_log_bands = bool(steerable_log_bands)

        # Progressive schedule. None => static weights, bit-identical to before.
        if steerable_band_schedule is not None and not isinstance(steerable_band_schedule, (list, tuple)):
            from omegaconf import OmegaConf
            steerable_band_schedule = OmegaConf.to_container(steerable_band_schedule, resolve=True)
        self.steerable_band_schedule = steerable_band_schedule
        self.steerable_band_ramp_epochs = int(steerable_band_ramp_epochs)
        self.steerable_band_schedule_order = str(steerable_band_schedule_order)
        self.steerable_normalize_by_active_levels = bool(steerable_normalize_by_active_levels)
        if self.steerable_band_schedule and self.steerable_band_weights is None:
            raise ValueError("steerable_band_schedule needs steerable_band_weights as its base")
        self._sched_cache = (None, None)   # (epoch, resolved dict)

        # Set by AutoencoderKLSteerableEMA each step. Plain int, never a buffer.
        self.current_epoch = 0

        # Plain attribute, never a buffer: nn.Module.__setattr__ only diverts a Tensor
        # into _buffers if the name is ALREADY there. So this never enters state_dict(),
        # LitEma's shadow, or init_from_ckpt's missing-key check.
        self.teacher_input = None

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split="train", weights=None):
        loss, log = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                                    global_step, last_layer=last_layer, cond=cond,
                                    split=split, weights=weights)

        if optimizer_idx != 0 or self.steerable_weight == 0.0 or global_step < self.steerable_start:
            return loss, log

        teacher = self.teacher_input
        if teacher is None:
            raise RuntimeError(
                "steerable term is enabled but no teacher was stashed. Use "
                "ldm_ftmse.autoencoder_steerable.AutoencoderKLSteerableEMA together with a "
                "ldm_ftmse.data_teacher.FileListSRTeacher* dataset.")
        if teacher.shape[0] != reconstructions.shape[0]:
            raise RuntimeError("teacher batch {} does not match reconstructions {}".format(
                tuple(teacher.shape), tuple(reconstructions.shape)))

        band_weights = self.steerable_band_weights
        if self.steerable_band_schedule:
            ep = int(self.current_epoch)
            if self._sched_cache[0] != ep:
                self._sched_cache = (ep, resolve_band_weights(
                    ep, self.steerable_band_weights,
                    height=self.steerable_height,
                    nbands=self.steerable_order + 1,
                    order=self.steerable_band_schedule_order,
                    schedule=self.steerable_band_schedule,
                    ramp_epochs=self.steerable_band_ramp_epochs,
                    normalize_by_active_levels=self.steerable_normalize_by_active_levels))
            band_weights = self._sched_cache[1]

        rel, per_band = compute_steerable_rel_l1(
            teacher.detach(), reconstructions,
            pyr_height=self.steerable_height,
            pyr_order=self.steerable_order,
            band_weights=band_weights,
            match_mode=self.steerable_match_mode,
            color_mode=self.steerable_color_mode,
            pyr_complex=self.steerable_complex,
            complex_metric=self.steerable_complex_metric,
            eps=self.steerable_eps,
            strict_band_weights=True,
        )

        scale = float(inputs[0].numel()) if self.steerable_normalize == "nll_units" else 1.0
        term = self.steerable_weight * scale * rel
        loss = loss + term

        log["{}/steerable_rel_l1".format(split)] = rel.detach()      # unit-free, comparable
        log["{}/steerable_term".format(split)] = term.detach()       # loss units
        log["{}/steerable_share".format(split)] = (term / loss).detach()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        if self.steerable_band_schedule:
            log["{}/steerable_active_levels".format(split)] = torch.tensor(
                float(active_levels_at(int(self.current_epoch), self.steerable_band_schedule,
                                       self.steerable_height)))
            log["{}/steerable_wsum".format(split)] = torch.tensor(
                float(sum(band_weights.values())))
        if self.steerable_log_bands:
            for name, v in per_band.items():
                log["{}/steer_band_{}".format(split, name)] = v.detach()
        return loss, log
