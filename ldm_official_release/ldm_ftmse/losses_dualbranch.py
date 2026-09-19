from __future__ import annotations
"""Loss routing for the dual-branch decoder (autoencoder_dualbranch.py).

THE POINT. The decoder now has two output tails, combined by a fixed frequency split
(band_split.py):

    out = LP(b1) + HP(b2)          LP = pyramid levels >= split_level + residual_lowpass

Upstream's loss takes ONE reconstruction and puts every term on it. Here each term is
pointed at the tensor it is supposed to train:

  mode "replace"   b1 is a full-image tail:  L1 + LPIPS on (GT, b1)
                   b2 owns the fine levels:  steerable rel-L1 on (teacher, b2) with every
                                             level >= split_level weighted 0
                   -> nothing sees `out` during training (there is no GAN in these arms),
                      which is why steerable_rel_l1_final and dual_final_l1 are logged.

  mode "split"     b1 is the low-pass image: L1 + LPIPS on (LP(GT), LP(b1))
                   the combined output:      steerable rel-L1 on (teacher, out), all bands

In both modes the discriminator -- if an arm ever turns one on -- sees `out`, and so does
`optimizer_idx == 1`, which falls straight through to the parent.

WHY A SUBCLASS. losses_steerable.py is untouched, on purpose: every existing arm, its
parity tests and its checkpoints keep the exact class they ran with. This class only adds
routing; the steerable machinery (_steer, _band_weights_now, the band schedule, the
sub-pixel head, nll_units scaling) is the parent's, reused as-is.

The consistency loop is refused rather than half-supported: it compares the teacher's
resampled roundtrip against the student, and which student tensor that should be in a
two-branch model is a separate design question.
"""

import torch

from . import band_split
from .band_schedule import active_levels_at, scale_level_groups
from .losses_steerable import LPIPSWithDiscriminatorSteerable
from taming.modules.losses.vqperceptual import adopt_weight

_DUAL_MODES = ("split", "replace")


class LPIPSDualBranchSteerable(LPIPSWithDiscriminatorSteerable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.steerable_consistency_loop:
            raise ValueError(
                "steerable_consistency_loop is not supported with the dual-branch decoder: "
                "which branch the loop's roundtrip should compare against is undefined.")
        # Set by the model (autoencoder_dualbranch). Plain attributes -- nn.Module only
        # diverts a Tensor into _buffers when the name is already registered -- so none of
        # this reaches state_dict(), LitEma's shadow or init_from_ckpt's missing-key check.
        self.dual_mode = None
        self.dual_split_level = 3
        self.dual_pad = 32
        # replace only: True gives branch 2 the steerable term on ALL bands (no zeroing of
        # levels >= split_level). Default False = the original replace routing, bit-identical.
        self.dual_steer_all_bands = False
        self.dual_branches = None

    # ------------------------------------------------------------------ helpers
    def _fine_band_weights(self, band_weights):
        """`band_weights` with every level >= dual_split_level zeroed.

        scale_level_groups already folds residual_highpass into level 0 and
        residual_lowpass into the coarsest level, so "level" here means the same thing it
        means to the band schedule. A weight of 0 resolves cleanly under
        strict_band_weights and drops out of the weighted mean sum(w*rel)/sum(w).
        """
        groups = scale_level_groups(self.steerable_height, self.steerable_order + 1,
                                    order="fine_to_coarse")
        if isinstance(band_weights, (list, tuple)):
            raise TypeError(
                "dual-branch 'replace' needs steerable_band_weights as a DICT keyed by band "
                "tuple (positional lists cannot be split by level)")

        def base_for(key):
            if band_weights is None:
                return 1.0
            for cand in (key, str(key)):
                if cand in band_weights:
                    return float(band_weights[cand])
            for dk, dv in band_weights.items():
                if str(dk) == str(key):
                    return float(dv)
            raise KeyError("no base weight for band {!r}".format(key))

        out = {}
        for li, keys in enumerate(groups):
            for k in keys:
                out[str(k)] = base_for(k) if li < int(self.dual_split_level) else 0.0
        if not any(v > 0.0 for v in out.values()):
            raise ValueError(
                "dual_branch_split_level={} leaves no active band for branch 2".format(
                    self.dual_split_level))
        return out

    def _lp(self, x):
        return band_split.lowpass(x, self.dual_split_level, self.dual_pad,
                                  order=self.steerable_order)

    def _gen_step_dual(self, inputs, reconstructions, posteriors, global_step,
                       last_layer, cond, split, weights, rec_inputs, rec_recon):
        """losses_steerable._gen_step (L420-486), carried with the pixel/LPIPS pair
        parameterised.

        `rec_inputs`/`rec_recon` feed the reconstruction terms; the GAN term and the KL
        term keep using `reconstructions` (= the combined output) and `posteriors`. Passing
        rec_inputs=inputs, rec_recon=reconstructions reproduces the parent bit-for-bit,
        which is what scripts/test_dual_branch.py pins.
        """
        diff = rec_inputs.contiguous() - rec_recon.contiguous()
        rec_loss = diff.pow(2) if self.pixel_loss == "l2" else diff.abs()
        rec_loss = self.pixel_weight * rec_loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(rec_inputs.contiguous(), rec_recon.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        if self.nll_reduction == "mean":
            weighted_nll_loss = torch.mean(weighted_nll_loss)
            nll_loss = torch.mean(nll_loss)
        else:
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        if self.disc_factor > 0.0:
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)
        else:
            g_loss = torch.zeros((), device=reconstructions.device)

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
        loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

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

    # ------------------------------------------------------------------ forward
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split="train", weights=None):
        if optimizer_idx != 0 or self.dual_mode is None:
            # Critic step (and any non-dual use): the parent's path, on the combined output.
            return super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                                   global_step, last_layer=last_layer, cond=cond,
                                   split=split, weights=weights)

        if self.dual_mode not in _DUAL_MODES:
            raise ValueError("dual_mode must be one of {}, got {!r}".format(_DUAL_MODES, self.dual_mode))
        if self.dual_branches is None:
            raise RuntimeError(
                "dual-branch loss is enabled but no branch outputs were stashed. Use "
                "ldm_ftmse.autoencoder_dualbranch.AutoencoderKLDualBranchEMANoGAN, whose "
                "decode() stashes (b1, b2) on the loss.")
        b1, b2 = self.dual_branches
        if b1.shape != reconstructions.shape:
            raise RuntimeError("stashed branch {} does not match reconstructions {}".format(
                tuple(b1.shape), tuple(reconstructions.shape)))

        # --- reconstruction terms -------------------------------------------------
        lp_b1 = None
        if self.dual_mode == "split":
            lp_b1 = self._lp(b1)
            rec_inputs, rec_recon = self._lp(inputs), lp_b1
        else:
            rec_inputs, rec_recon = inputs, b1
        loss, log = self._gen_step_dual(inputs, reconstructions, posteriors, global_step,
                                        last_layer, cond, split, weights,
                                        rec_inputs=rec_inputs, rec_recon=rec_recon)

        # --- diagnostics on the tensor that is actually deployed ------------------
        with torch.no_grad():
            log["{}/dual_final_l1".format(split)] = (inputs - reconstructions).abs().mean().detach()
            lp_ref = lp_b1 if lp_b1 is not None else self._lp(b1)
            hp = reconstructions - lp_ref                      # = HP(b2), the fine half
            log["{}/dual_b2_share".format(split)] = (
                hp.float().pow(2).mean() / reconstructions.float().pow(2).mean().clamp_min(1e-12)).detach()

        usual_on = self.steerable_weight != 0.0
        if global_step < self.steerable_start or not usual_on:
            return loss, log

        teacher = self.teacher_input
        if teacher is None:
            raise RuntimeError(
                "steerable term is enabled but no teacher was stashed. Use the dual-branch "
                "autoencoder together with a ldm_ftmse.data_teacher/data_nerc teacher dataset.")
        if teacher.shape[0] != reconstructions.shape[0]:
            raise RuntimeError("teacher batch {} does not match reconstructions {}".format(
                tuple(teacher.shape), tuple(reconstructions.shape)))

        # --- steerable term -------------------------------------------------------
        band_weights = self._band_weights_now()
        scale = float(inputs[0].numel()) if self.steerable_normalize == "nll_units" else 1.0
        if self.dual_mode == "replace":
            steer_weights = (band_weights if self.dual_steer_all_bands
                             else self._fine_band_weights(band_weights))
            steer_target = b2                     # branch 2 IS the fine half
        else:
            steer_weights = band_weights
            steer_target = reconstructions        # the combined output
        rel, per_band = self._steer(teacher, steer_target, steer_weights)
        term = self.steerable_weight * scale * rel
        norm_factor = 1.0
        if self.steerable_normalize_by_active_levels and self.steerable_band_schedule:
            n_act = active_levels_at(int(self.current_epoch), self.steerable_band_schedule,
                                     self.steerable_height)
            if n_act > 0:
                norm_factor = 1.0 / float(n_act)
                term = term * norm_factor
        loss = loss + term

        log["{}/steerable_rel_l1".format(split)] = rel.detach()
        log["{}/steerable_term".format(split)] = term.detach()
        if self.steerable_band_schedule:
            log["{}/steerable_active_levels".format(split)] = torch.tensor(
                float(active_levels_at(int(self.current_epoch), self.steerable_band_schedule,
                                       self.steerable_height)))
            log["{}/steerable_wsum".format(split)] = torch.tensor(float(sum(steer_weights.values())))
            log["{}/steerable_norm_factor".format(split)] = torch.tensor(norm_factor)
        if self.steerable_log_bands:
            for name, v in per_band.items():
                log["{}/steer_band_{}".format(split, name)] = v.detach()
        if self.up_head is not None:
            log["{}/steer_up_head_drift".format(split)] = self.up_head.drift()

        # The trained rel-L1 above is branch 2 on fine bands only (replace) -- not
        # comparable with any other arm. Log the usual full-weight number on the deployed
        # output too, exactly as the consistency loop logs its untrained term.
        if self.dual_mode == "replace":
            with torch.no_grad():
                rel_final, _ = self._steer(teacher, reconstructions.detach(), band_weights)
            log["{}/steerable_rel_l1_final".format(split)] = rel_final.detach()

        log["{}/steerable_share".format(split)] = (term / loss).detach()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        return loss, log
