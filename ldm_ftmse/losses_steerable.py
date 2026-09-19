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

CONSISTENCY LOOP (`steerable_consistency_loop: True`). The usual term compares
the native 512 teacher T against the bicubic-upsampled 256 student up(S). That
asks the student for content above its own Nyquist -- residual_highpass and
band_0 at 512 sit there, and every upsample run has watched them stay pinned.
The loop instead pushes the TEACHER through the same resampling path the student
is judged through, and compares like with like:

    down stage      down(T) at 256      vs  S      at 256
    roundtrip stage up(down(T)) at 512  vs  up(S)  at 512

`down` is the bicubic used by match_mode "downsample"; `up` is the configured
student-side upsampler (see UPSAMPLER below; bicubic by default). Both teacher
tensors are built under no_grad, so the loop itself trains nothing new.
`steerable_consistency_usual_term` keeps or drops the original T-vs-up(S) term
alongside the loop. When it is dropped, `steerable_rel_l1` is still LOGGED
(under no_grad, one extra pyramid forward) so the metric stays comparable
across arms.

UPSAMPLER (`steerable_upsampler`). "bicubic" (default) is the fixed
F.interpolate every run so far used; the checkpoint is byte-identical to before.
"subpixel" replaces it with ldm_ftmse.subpixel_up.SubPixelUp2x -- a learned,
linear conv + PixelShuffle(2) polyphase filter (100 parameters) initialised to
reproduce bicubic exactly, so step 0 is the bicubic arm and the head then trains
in opt_ae alongside the decoder (autoencoder.py configure_optimizers, at
lr * steerable_upsampler_lr_mult). The head lives on the loss as `up_head`, so
its two tensors sit under `loss.` in state_dict (init_from_ckpt tolerates that),
get an EMA shadow like the discriminator, and are never part of the deployed
decoder. `{split}/steer_up_head_drift` = max |weight - bicubic init| says whether
it moved. The consistency loop's `up` goes through the same head under no_grad.
The down direction stays bicubic: pixel shuffle cannot downsample.
"""

import torch
import torch.nn.functional as F

from ldm.modules.losses.contperceptual import LPIPSWithDiscriminator
# _disc_only carries upstream's optimizer_idx==1 branch, which uses adopt_weight.
# contperceptual gets it via `from taming...vqperceptual import *`; import it
# explicitly here rather than relying on that star-import leaking through.
from taming.modules.losses.vqperceptual import adopt_weight

from .band_schedule import active_levels_at, resolve_band_weights
from .steerable_rel_l1 import (_max_steerable_pyr_height_for_size,
                               compute_steerable_rel_l1)
from .subpixel_up import SubPixelUp2x

_LOOP_STAGES = ("down", "roundtrip", "both")
_UPSAMPLERS = ("bicubic", "subpixel")


def shift_band_weights_to_half_res(band_weights, *, pyr_height, full_side, half_side):
    """Re-key a full-resolution band-weight dict for a pyramid built at half the side.

    Band indices are grid-relative: level 0 is always the octave just under the
    grid's Nyquist. Halving the side halves the Nyquist, so ('band', l, b) on the
    half grid covers the SAME cycles-per-image as ('band', l+1, b) on the full grid.
    A weight profile is a statement about physical frequencies (the graded ramp
    weights 16-64 cpi, where identity lives, over 128-256 cpi), so it is carried
    across by physical frequency, not by index:

        half ('band', l, b)       <- full ('band', l+1, b)
        half residual_highpass    <- mean of full ('band', 0, *)   (same octave)
        half residual_lowpass     <- full residual_lowpass          (same cutoff)
        full residual_highpass    -> dropped (above the half grid's Nyquist)

    Only defined when the half grid has exactly one level fewer than the full
    grid -- true for height 5 at 512/256 (22 -> 18 bands). Otherwise the lowpass
    cutoffs differ and no index shift is honest; pass
    `steerable_consistency_down_band_weights` explicitly instead.
    """
    if band_weights is None:
        return None
    if not isinstance(band_weights, dict):
        raise TypeError(
            "the consistency loop's down stage can only derive weights from a DICT of "
            "band weights; got {}. Give steerable_consistency_down_band_weights "
            "explicitly.".format(type(band_weights).__name__))
    h_full = min(int(pyr_height) + 2, _max_steerable_pyr_height_for_size(full_side)) - 2
    h_half = min(int(pyr_height) + 2, _max_steerable_pyr_height_for_size(half_side)) - 2
    if h_half != h_full - 1:
        raise ValueError(
            "cannot shift band weights from {}px ({} levels) to {}px ({} levels): the "
            "half grid must have exactly one level fewer. Give "
            "steerable_consistency_down_band_weights explicitly.".format(
                full_side, h_full, half_side, h_half))

    by_str = {str(k): float(v) for k, v in band_weights.items()}
    out, used = {}, set()

    def take(key):
        sk = str(key)
        if sk not in by_str:
            raise KeyError("no full-res weight for band {!r}".format(key))
        used.add(sk)
        return by_str[sk]

    finest = [take(("band", 0, b)) for b in range(sum(1 for k in by_str if k.startswith("('band', 0,")))]
    if not finest:
        raise KeyError("no ('band', 0, *) weights to fold into the half-res residual_highpass")
    out[str(("residual_highpass",))] = sum(finest) / len(finest)
    for l in range(h_half):
        b = 0
        while str(("band", l + 1, b)) in by_str:
            out[str(("band", l, b))] = take(("band", l + 1, b))
            b += 1
        if b == 0:
            raise KeyError("no ('band', {}, *) weights in full-res dict".format(l + 1))
    out[str(("residual_lowpass",))] = take(("residual_lowpass",))
    used.add(str(("residual_highpass",)))          # dropped on purpose
    leftover = sorted(set(by_str) - used)
    if leftover:
        raise ValueError("full-res band weights never mapped to the half grid: {}".format(leftover))
    return out


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
                 steerable_complex_metric_overrides=None,
                 steerable_eps=1e-8,
                 steerable_mag_pool=False,                # ON/OFF: box-pool band magnitudes
                 steerable_mag_pool_kernel=3,             # pool size when on (odd)
                 steerable_start=0,
                 steerable_log_bands=False,
                 steerable_band_schedule=None,
                 steerable_band_ramp_epochs=0,
                 steerable_band_schedule_order="coarse_to_fine",
                 steerable_normalize_by_active_levels=False,
                 # --- consistency loop (see module docstring) --------------------
                 steerable_consistency_loop=False,
                 steerable_consistency_usual_term=True,
                 steerable_consistency_stages="both",     # "down" | "roundtrip" | "both"
                 steerable_consistency_weight=None,       # None -> steerable_weight
                 steerable_consistency_down_band_weights=None,  # None -> shifted from base
                 # --- student-side upsampler (see module docstring) ----------------
                 steerable_upsampler="bicubic",           # "bicubic" | "subpixel"
                 steerable_upsampler_normalize_dc=True,   # subpixel: taps sum to 1 per phase
                 steerable_upsampler_lr_mult=1.0,         # subpixel: head lr = lr * this
                 disc_skip_recon_loss=False,
                 pixel_loss="l1",
                 nll_reduction="sum",                     # "sum" = upstream | "mean"
                 perceptual_net="vgg",                    # "vgg" = upstream | "alex" (nerc's default)
                 adaptive_weight_grad_scale=1.0,          # >1: fp16 guard for d_weight (see method)
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.disc_skip_recon_loss = bool(disc_skip_recon_loss)

        # Upstream hardcodes L1 and never reads the `pixelloss_weight` it stores.
        # sd-vae-ft-mse's published recipe is MSE + 0.1*LPIPS, so both need to be
        # reachable. The carried generator branch below runs ONLY when one of these
        # is non-default, so every existing arm keeps calling upstream unchanged.
        if pixel_loss not in ("l1", "l2"):
            raise ValueError("pixel_loss must be 'l1' or 'l2'")
        self.pixel_loss = pixel_loss
        # Upstream reduces nll as a per-image SUM over C*H*W; "mean" divides that by
        # C*H*W as well (what distillation_nerc and most reimplementations do).
        # perceptual_net: "alex" swaps upstream's VGG16 LPIPS for AlexNet LPIPS
        # (ldm_ftmse/lpips_alex.py, bit-identical to lpips.LPIPS(net="alex") -- what
        # distillation_nerc builds by default, lpips_net="alex"). Same call signature and
        # [N,1,1,1] output, so every line that uses self.perceptual_loss is unchanged.
        if perceptual_net not in ("vgg", "alex"):
            raise ValueError("perceptual_net must be 'vgg' or 'alex'")
        self.perceptual_net = perceptual_net
        if perceptual_net == "alex":
            from .lpips_alex import LPIPSAlex
            self.perceptual_loss = LPIPSAlex().eval()
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False
            print("[steerable] perceptual trunk = AlexNet LPIPS (was VGG16)")
        self.adaptive_weight_grad_scale = float(adaptive_weight_grad_scale)
        if nll_reduction not in ("sum", "mean"):
            raise ValueError("nll_reduction must be 'sum' or 'mean'")
        self.nll_reduction = nll_reduction
        self._custom_pixel = ((pixel_loss != "l1") or (float(self.pixel_weight) != 1.0)
                              or nll_reduction != "sum")

        # OmegaConf hands us a DictConfig, which is NOT a dict -- _resolve_band_weights
        # would hit its TypeError. Convert once, here.
        if steerable_band_weights is not None and not isinstance(steerable_band_weights, (dict, list, tuple)):
            from omegaconf import OmegaConf
            steerable_band_weights = OmegaConf.to_container(steerable_band_weights, resolve=True)

        if steerable_normalize not in ("nll_units", "none"):
            raise ValueError("steerable_normalize must be 'nll_units' or 'none'")

        # mean-reduced nll + nll_units would multiply the steerable term by C*H*W
        # while nll is divided by it: the term becomes ~99.999% of the objective
        # (measured). The two unit systems must match, so forbid the mix outright.
        if nll_reduction == "mean" and steerable_normalize == "nll_units" and float(steerable_weight) != 0.0:
            raise ValueError("nll_reduction='mean' requires steerable_normalize='none': "
                             "nll_units scales the term by C*H*W for a SUM-reduced nll")
        self.steerable_weight = float(steerable_weight)
        self.steerable_normalize = steerable_normalize
        self.steerable_height = int(steerable_height)
        self.steerable_order = int(steerable_order)
        self.steerable_band_weights = steerable_band_weights
        self.steerable_match_mode = steerable_match_mode
        self.steerable_color_mode = steerable_color_mode
        self.steerable_complex = bool(steerable_complex)
        self.steerable_complex_metric = steerable_complex_metric
        # per-band metric map (see steerable_rel_l1.complex_metric_overrides); None = global metric
        self.steerable_complex_metric_overrides = (
            dict(steerable_complex_metric_overrides) if steerable_complex_metric_overrides else None)
        self.steerable_eps = float(steerable_eps)
        # Pooled-magnitude comparison (steerable_rel_l1.mag_pool_kernel). OFF by default,
        # which is the original pointwise code path; the kernel only matters when on.
        self.steerable_mag_pool = bool(steerable_mag_pool)
        self.steerable_mag_pool_kernel = int(steerable_mag_pool_kernel) if self.steerable_mag_pool else 0
        if self.steerable_mag_pool and self.steerable_mag_pool_kernel < 3:
            raise ValueError("steerable_mag_pool: true needs steerable_mag_pool_kernel >= 3 (odd)")
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

        # --- consistency loop ---------------------------------------------------
        self.steerable_consistency_loop = bool(steerable_consistency_loop)
        self.steerable_consistency_usual_term = bool(steerable_consistency_usual_term)
        self.steerable_consistency_stages = str(steerable_consistency_stages).strip().lower()
        self.steerable_consistency_weight = (self.steerable_weight if steerable_consistency_weight is None
                                             else float(steerable_consistency_weight))
        if steerable_consistency_down_band_weights is not None and not isinstance(
                steerable_consistency_down_band_weights, (dict, list, tuple)):
            from omegaconf import OmegaConf
            steerable_consistency_down_band_weights = OmegaConf.to_container(
                steerable_consistency_down_band_weights, resolve=True)
        self.steerable_consistency_down_band_weights = steerable_consistency_down_band_weights
        if self.steerable_consistency_loop:
            if self.steerable_consistency_stages not in _LOOP_STAGES:
                raise ValueError("steerable_consistency_stages must be one of {}, got {!r}".format(
                    _LOOP_STAGES, steerable_consistency_stages))
            if str(self.steerable_match_mode).strip().lower() != "upsample":
                raise ValueError(
                    "steerable_consistency_loop needs steerable_match_mode 'upsample': the loop "
                    "is defined as the teacher taking the student's up-path. With 'downsample' "
                    "the usual term already IS the loop's down stage.")
            if not self.steerable_consistency_usual_term and self.steerable_consistency_weight == 0.0:
                raise ValueError("consistency loop on, usual term off, loop weight 0: nothing to train")
            print("[steerable] consistency loop ON: stages={} weight={} usual_term={} "
                  "down_band_weights={}".format(
                      self.steerable_consistency_stages, self.steerable_consistency_weight,
                      self.steerable_consistency_usual_term,
                      "explicit" if self.steerable_consistency_down_band_weights is not None
                      else "shifted from steerable_band_weights"))

        # --- student-side upsampler ----------------------------------------------
        self.steerable_upsampler = str(steerable_upsampler).strip().lower()
        self.steerable_upsampler_lr_mult = float(steerable_upsampler_lr_mult)
        if self.steerable_upsampler not in _UPSAMPLERS:
            raise ValueError("steerable_upsampler must be one of {}, got {!r}".format(
                _UPSAMPLERS, steerable_upsampler))
        if self.steerable_upsampler == "subpixel":
            if str(self.steerable_match_mode).strip().lower() != "upsample":
                raise ValueError(
                    "steerable_upsampler 'subpixel' needs steerable_match_mode 'upsample': the "
                    "head lifts the STUDENT 256 -> 512. With 'downsample' nothing is upsampled "
                    "and the head would be dead weight.")
            # A real submodule: in state_dict under loss.up_head.*, EMA-shadowed,
            # moved to device with the model, trained via configure_optimizers.
            self.up_head = SubPixelUp2x(normalize_dc=bool(steerable_upsampler_normalize_dc))
            print("[steerable] upsampler: subpixel (conv5x5 + PixelShuffle, {} params, "
                  "bicubic init, normalize_dc={}, lr_mult={})".format(
                      self.up_head.weight.numel(), self.up_head.normalize_dc,
                      self.steerable_upsampler_lr_mult))
        else:
            self.up_head = None

        # Set by AutoencoderKLSteerableEMA each step. Plain int, never a buffer.
        self.current_epoch = 0

        # Plain attribute, never a buffer: nn.Module.__setattr__ only diverts a Tensor
        # into _buffers if the name is ALREADY there. So this never enters state_dict(),
        # LitEma's shadow, or init_from_ckpt's missing-key check.
        self.teacher_input = None

    def _disc_only(self, inputs, reconstructions, global_step, cond, split):
        """contperceptual.py:94-110, carried verbatim, minus the work above it.

        Upstream computes rec_loss, the LPIPS VGG forward and posteriors.kl() on
        lines 48-60 for BOTH optimizer indices -- and the optimizer_idx==1 branch
        below uses NONE of them: d_loss depends only on the two discriminator
        logits, and the log dict it returns carries only those. So for the disc
        step that entire prefix is computed and discarded (~47 ms of LPIPS alone,
        of a ~1543 ms step).

        This returns the same d_loss and the same log keys, so it is BIT-EXACT.
        It is a carried copy of upstream, which is why test_faststep_parity.py
        pins it against the real thing -- the same treatment losses.py gets.
        `posteriors` is not a parameter here precisely because it is unused;
        that is what lets disc_reuse_recon skip the forward entirely.
        """
        if cond is None:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
        else:
            logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
            logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

        disc_factor = adopt_weight(self.disc_factor, global_step,
                                   threshold=self.discriminator_iter_start)
        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

        log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
               "{}/logits_real".format(split): logits_real.detach().mean(),
               "{}/logits_fake".format(split): logits_fake.detach().mean()}
        return d_loss, log

    def _band_weights_now(self):
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
                    normalize_by_active_levels=False))   # applied to the TERM instead -- see forward
            band_weights = self._sched_cache[1]
        return band_weights

    def _steer(self, teacher, student, band_weights):
        t_size = teacher.shape[-2:]
        # None keeps steerable_rel_l1's own bicubic path, bit-identical to before.
        up_fn = (lambda s: self._up(s, t_size)) if self.up_head is not None else None
        return compute_steerable_rel_l1(
            teacher.detach(), student,
            pyr_height=self.steerable_height,
            pyr_order=self.steerable_order,
            band_weights=band_weights,
            match_mode=self.steerable_match_mode,
            color_mode=self.steerable_color_mode,
            pyr_complex=self.steerable_complex,
            complex_metric=self.steerable_complex_metric,
            complex_metric_overrides=self.steerable_complex_metric_overrides,
            eps=self.steerable_eps,
            mag_pool_kernel=self.steerable_mag_pool_kernel,
            strict_band_weights=True,
            upsample_fn=up_fn,
        )

    @staticmethod
    def _down(x, size):
        # The bicubic match_mode "downsample" uses (steerable_rel_l1.py). Down is
        # ALWAYS bicubic: the sub-pixel head only goes up.
        return F.interpolate(x, size=size, mode="bicubic", align_corners=False)

    def _up(self, x, size):
        """Student-side upsampler: the configured head, else the bicubic match_mode
        "upsample" uses. The loop's whole point is that the teacher's roundtrip goes
        through exactly the same `up` the student is judged through."""
        size = tuple(int(s) for s in size)
        if self.up_head is None:
            return F.interpolate(x, size=size, mode="bicubic", align_corners=False)
        want = (2 * int(x.shape[-2]), 2 * int(x.shape[-1]))
        if size != want:
            raise ValueError(
                "SubPixelUp2x is a fixed 2x head: input {} can only go to {}, not {}. "
                "Use teacher_size = 2 * size or steerable_upsampler: bicubic.".format(
                    tuple(x.shape), want, size))
        return self.up_head(x)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """Upstream's d_weight = |grad nll| / (|grad g| + 1e-4), optionally computed from
        the gradients of S*nll and S*g with the epsilon scaled by S -- identical in exact
        arithmetic: (S a) / (S b + S 1e-4) = a / (b + 1e-4).

        Why: torch.autograd.grad is NOT loss-scaled by AMP. Under precision=16 with a
        MEAN-reduced nll the gradient reaching each output pixel is ~2*0.1/(12*196608)
        ~ 8e-8, below fp16's normal range, so |grad nll| collapses and d_weight -- and with
        it the whole GAN term -- goes silently to ~0. S lifts those gradients into range.
        S = 1 (default) is upstream's method verbatim.
        """
        S = self.adaptive_weight_grad_scale
        if S == 1.0 or last_layer is None:
            return super().calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
        nll_grads = torch.autograd.grad(nll_loss * S, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss * S, last_layer, retain_graph=True)[0]
        d_weight = torch.norm(nll_grads.float()) / (torch.norm(g_grads.float()) + 1e-4 * S)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def _gen_step(self, inputs, reconstructions, posteriors, global_step,
                  last_layer, cond, split, weights):
        """contperceptual.py:48-92 carried, with only the pixel term parameterised.

        Everything else -- logvar, the kl term, the adaptive d_weight, adopt_weight
        and every log key -- is upstream verbatim, so with pixel_loss='l1' and
        pixelloss_weight=1.0 this is bit-identical to super().forward(). It is only
        reached when one of those is non-default; test_loss_parity.py pins it the
        same way _disc_only is pinned.
        """
        diff = inputs.contiguous() - reconstructions.contiguous()
        rec_loss = diff.pow(2) if self.pixel_loss == "l2" else diff.abs()
        rec_loss = self.pixel_weight * rec_loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
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
        kl_loss = posteriors.kl()                  # already per-image, shape [N]
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
            # GAN off: skip the critic forward entirely (its term is 0 * g_loss anyway)
            g_loss = torch.zeros((), device=reconstructions.device)

        if self.disc_factor > 0.0:
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss,
                                                          last_layer=last_layer)
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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split="train", weights=None):
        if optimizer_idx == 1 and self.disc_skip_recon_loss:
            return self._disc_only(inputs, reconstructions, global_step, cond, split)

        if optimizer_idx == 0 and self._custom_pixel:
            loss, log = self._gen_step(inputs, reconstructions, posteriors, global_step,
                                       last_layer, cond, split, weights)
        else:
            loss, log = super().forward(inputs, reconstructions, posteriors, optimizer_idx,
                                        global_step, last_layer=last_layer, cond=cond,
                                        split=split, weights=weights)

        loop_on = self.steerable_consistency_loop and self.steerable_consistency_weight != 0.0
        usual_on = self.steerable_weight != 0.0 and (
            self.steerable_consistency_usual_term or not self.steerable_consistency_loop)
        if optimizer_idx != 0 or global_step < self.steerable_start or not (usual_on or loop_on):
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

        band_weights = self._band_weights_now()
        scale = float(inputs[0].numel()) if self.steerable_normalize == "nll_units" else 1.0

        # --- usual term: T (512) vs up(S) ------------------------------------------
        if usual_on:
            rel, per_band = self._steer(teacher, reconstructions, band_weights)
        else:
            with torch.no_grad():           # logged for comparability, not trained
                rel, per_band = self._steer(teacher, reconstructions.detach(), band_weights)
        term = self.steerable_weight * scale * rel if usual_on else torch.zeros_like(rel)
        # nerc semantics: divide the steerable term by the number of ACTIVE scale
        # levels, so it shrinks as a fine_to_coarse schedule switches levels on.
        # (Applied to the weights it would cancel in sum(w*rel)/sum(w) -- a no-op.)
        norm_factor = 1.0
        if self.steerable_normalize_by_active_levels and self.steerable_band_schedule:
            n_act = active_levels_at(int(self.current_epoch), self.steerable_band_schedule,
                                     self.steerable_height)
            if n_act > 0:
                norm_factor = 1.0 / float(n_act)
                term = term * norm_factor
        loss = loss + term

        log["{}/steerable_rel_l1".format(split)] = rel.detach()      # unit-free, comparable
        log["{}/steerable_term".format(split)] = term.detach()       # loss units
        if self.steerable_band_schedule:
            log["{}/steerable_active_levels".format(split)] = torch.tensor(
                float(active_levels_at(int(self.current_epoch), self.steerable_band_schedule,
                                       self.steerable_height)))
            log["{}/steerable_wsum".format(split)] = torch.tensor(
                float(sum(band_weights.values())))
            log["{}/steerable_norm_factor".format(split)] = torch.tensor(norm_factor)
        if self.steerable_log_bands:
            for name, v in per_band.items():
                log["{}/steer_band_{}".format(split, name)] = v.detach()
        if self.up_head is not None:
            log["{}/steer_up_head_drift".format(split)] = self.up_head.drift()

        # --- consistency loop: teacher takes the student's resampling path ---------
        loop_term = torch.zeros_like(rel)
        if loop_on:
            t_size, s_size = teacher.shape[-2:], reconstructions.shape[-2:]
            if tuple(t_size) == tuple(s_size):
                raise RuntimeError(
                    "consistency loop is degenerate: teacher and student are both {} so "
                    "down/up are identities. It needs a teacher above the student's "
                    "resolution (teacher_size 512 for a 256 student).".format(tuple(t_size)))
            stages = self.steerable_consistency_stages
            with torch.no_grad():
                teacher_dn = self._down(teacher, s_size)                     # 512 -> 256, bicubic
                teacher_rt = (self._up(teacher_dn, t_size)                    # 256 -> 512, same up as S
                              if stages in ("roundtrip", "both") else None)

            rels = []
            if stages in ("down", "both"):
                dn_weights = self.steerable_consistency_down_band_weights
                if dn_weights is None:
                    dn_weights = shift_band_weights_to_half_res(
                        band_weights, pyr_height=self.steerable_height,
                        full_side=min(t_size), half_side=min(s_size))
                # Same size on both sides, so match_mode does nothing: compared at 256.
                rel_dn, per_band_dn = self._steer(teacher_dn, reconstructions, dn_weights)
                rels.append(rel_dn)
                log["{}/steer_cl_down_rel_l1".format(split)] = rel_dn.detach()
                if self.steerable_log_bands:
                    for name, v in per_band_dn.items():
                        log["{}/steer_cl_down_band_{}".format(split, name)] = v.detach()
            if stages in ("roundtrip", "both"):
                # up(down(T)) at 512 vs up(S): the student is upsampled inside by the
                # same `_up` (bicubic or the sub-pixel head) that produced teacher_rt.
                rel_rt, per_band_rt = self._steer(teacher_rt, reconstructions, band_weights)
                rels.append(rel_rt)
                log["{}/steer_cl_rt_rel_l1".format(split)] = rel_rt.detach()
                if self.steerable_log_bands:
                    for name, v in per_band_rt.items():
                        log["{}/steer_cl_rt_band_{}".format(split, name)] = v.detach()

            # MEAN over active stages, so "both" carries the same weight as one stage
            # and steerable_consistency_weight reads in rel-L1 units either way.
            rel_loop = torch.stack(rels).mean()
            loop_term = self.steerable_consistency_weight * scale * rel_loop
            loss = loss + loop_term
            log["{}/steer_cl_rel_l1".format(split)] = rel_loop.detach()
            log["{}/steer_cl_term".format(split)] = loop_term.detach()

        # steerable_share is the TOTAL distillation share (usual + loop) -- the number
        # the share ladder is read in. Loop-free runs: identical to before.
        distill = term + loop_term
        log["{}/steerable_share".format(split)] = (distill / loss).detach()
        if loop_on:
            log["{}/steer_cl_share".format(split)] = (loop_term / loss).detach()
        log["{}/total_loss".format(split)] = loss.clone().detach().mean()
        return loss, log
