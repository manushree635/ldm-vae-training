from __future__ import annotations
"""Band-balanced MSE: the objective that recovered the fine bands in the gate probes, as an ldm loss.

distillation_nerc/analysis/stage_ledger/gate_probe.py (2026-09-17) trained encoder/decoder variants under
exactly this loss and scored them with a phase-SENSITIVE per-band correlation. It is what moved L1 (45-91
cycles/image at 256) from 0.517 to 0.658 with the whole encoder training -- where the magnitude-only steerable
rel-L1 driving the encoder pushed it DOWN to ~0.40. So this is a phase-sensitive band loss, unlike strpy.

TERM. The 256 output and the target are each split (per RGB channel) into the 6 levels of the training
steerable pyramid built on the 256 frame at height 6 (lo, L4, L3, L2, L1, L0*: L1..L4 = the 512-frame loss's
levels, L0* = the part of the 512 L0 a 256 image can hold). Each level is an exact LTI filter, applied as an
FFT mask (reflect-padded to 384, as steer_bands.py does). Per level: MSE over the image minus an 8 px border,
divided by that level's mean ENERGY in the training data, then averaged over the 6 levels -- so every band
counts equally however little energy it carries (the fine bands carry ~1/200 of the coarse one).

    bandbal = mean_levels( MSE_level(pred, target) / E_level )
    term    = bandbal_weight * (3*256*256 if nll_units) * bandbal       # same units as the steerable term

TARGET (`bandbal_target`) and GEOMETRY (`bandbal_geometry`):
    gt,      256  the ground-truth 256 input (the probe's winner)
    teacher, 256  the stashed 512 teacher recon, area-downsampled to 256 (the probe found this ~0.02 L1 worse,
                  concentrated in L0*)
    teacher, 512  the 256 reconstruction UPSAMPLED to 512 by the parent's `_up` (bicubic, or the sub-pixel head
                  if configured -- the same path the steerable term judges the student through) and compared with
                  the 512 teacher on the 512 frame: pyramid height 7 -> levels lo, L4, L3, L2, L1, L0, hi
                  (hi = 182-362 cpi, L0 = 91-181; the 256 frame's L0* is the part of L0 below 128).
    `bandbal_levels` picks which levels count (default: all of the frame's). hi at 512 lies entirely above the
    256 student's Nyquist, so its term is unreachable and only asks for invented energy (the dual-branch arm's
    Nyquist pressure); the 512 configs therefore list the levels explicitly without it.

ENERGIES (`bandbal_energies`): a list of 6 floats (lo..L0*), or null = measured on the first
`bandbal_energy_batches` generator batches of the TARGET, averaged over DDP ranks, then frozen. While
measuring, the running mean so far is used. They are plain attributes, never buffers, so the state_dict key set
stays identical to every other arm (exports and init_from_ckpt untouched).

The steerable term of the parent is expected OFF (steerable_weight 0): with it off the parent's forward returns
straight after upstream's L1 + LPIPS + GAN step (plus the bit-exact disc_skip_recon_loss path), and this class
adds the band term on the generator step only. The discriminator step is untouched. The GAN's adaptive weight
is computed inside upstream from nll_loss alone, so -- exactly like the steerable term -- this term does not
move d_weight.
"""

import torch
import torch.nn.functional as F

from .losses_steerable import LPIPSWithDiscriminatorSteerable
from .steerable_rel_l1 import _get_steerable_pyr

LEVELS = ("lo", "L4", "L3", "L2", "L1", "L0*")      # 256 frame, coarse -> fine
LEVELS_512 = ("lo", "L4", "L3", "L2", "L1", "L0", "hi")
_FRAMES = {256: dict(height=6, names=LEVELS), 512: dict(height=7, names=LEVELS_512)}


def _geom(side):
    """(padded FFT size, pad, border) for a frame: pad side/4 each way, border side/32 -- 384/64/8 at 256."""
    return side * 3 // 2, side // 4, side // 32


def build_level_masks(device="cpu", side=256):
    """(K, P0, P0/2+1) rfft masks of the steerable pyramid levels on a `side` frame (steer_bands.build).

    256 frame: height 6 -> lo, L4, L3, L2, L1, L0*.  512 frame: height 7 -> lo, L4, L3, L2, L1, L0, hi.
    Same physical bands (cycles/image) on both frames: 512's L1..L4 and lo are 256's L1..L4 and lo."""
    height, names = _FRAMES[side]["height"], _FRAMES[side]["names"]
    P0 = _geom(side)[0]
    pyr = _get_steerable_pyr(height=height, nbands=4, scale_factor=2, device=device)
    dlt = torch.zeros(1, 1, P0, P0, device=device)
    dlt[..., P0 // 2, P0 // 2] = 1
    coeff = pyr.build(dlt)
    nlev = len(coeff) - 2

    def only(cf, i):
        out, k = [], 0
        for c in cf:
            if isinstance(c, list):
                out.append([b if k + j == i else torch.zeros_like(b) for j, b in enumerate(c)])
                k += len(c)
            else:
                out.append(c if k == i else torch.zeros_like(c))
                k += 1
        return out

    D = torch.fft.fft2(dlt[0, 0].double())
    H = torch.stack([(torch.fft.fft2(pyr.reconstruct(only(coeff, i))[0].double()) / D).real
                     for i in range(2 + 4 * nlev)])
    if float((H.sum(0) - 1).abs().max()) >= 1e-3:
        raise RuntimeError("steerable level masks do not sum to 1 -- pyramid changed?")
    if side == 256:      # highpass residual = L0*, then band levels L1.. (steer_bands.py's labelling)
        groups = {"L0*": [0], "lo": [1 + 4 * nlev]}
        for lv in range(nlev):
            groups[f"L{lv + 1}"] = list(range(1 + 4 * lv, 5 + 4 * lv))
    else:                # highpass residual = hi, then band levels L0.. (the 512-frame training loss's labelling)
        groups = {"hi": [0], "lo": [1 + 4 * nlev]}
        for lv in range(nlev):
            groups[f"L{lv}"] = list(range(1 + 4 * lv, 5 + 4 * lv))
    assert set(groups) == set(names), (sorted(groups), names)
    HR = H[..., :P0 // 2 + 1].float()
    return torch.stack([HR[groups[n]].sum(0) for n in names])


def level_filter(x, masks):
    """x (B,C,S,S) -> (K,B,C,S,S), fp32, reflect-padded FFT filtering on the frame S the masks were built for."""
    side = int(x.shape[-1])
    P0, pad, _ = _geom(side)
    if masks.shape[-2] != P0:
        raise ValueError(f"masks are for a {masks.shape[-2]}-point FFT, input side {side} needs {P0}")
    xp = F.pad(x.float(), (pad,) * 4, mode="reflect")
    X = torch.fft.rfft2(xp)
    y = torch.fft.irfft2(X[None] * masks[:, None, None], s=(P0, P0))
    return y[..., pad:pad + side, pad:pad + side]


class LPIPSWithDiscriminatorBandBal(LPIPSWithDiscriminatorSteerable):

    def __init__(self, *args, bandbal_weight=1.0, bandbal_target="gt", bandbal_normalize="nll_units",
                 bandbal_energies=None, bandbal_energy_batches=16, bandbal_geometry=256, bandbal_levels=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if bandbal_target not in ("gt", "teacher"):
            raise ValueError(f"bandbal_target must be 'gt' or 'teacher', got {bandbal_target!r}")
        if bandbal_normalize not in ("nll_units", "none"):
            raise ValueError(f"bandbal_normalize must be 'nll_units' or 'none', got {bandbal_normalize!r}")
        if self.steerable_weight != 0.0:
            print(f"[bandbal] NOTE: steerable_weight={self.steerable_weight} -- the steerable term is ALSO on")
        self.bandbal_weight = float(bandbal_weight)
        self.bandbal_target = bandbal_target
        self.bandbal_normalize = bandbal_normalize
        self.bandbal_energy_batches = int(bandbal_energy_batches)
        self.bandbal_geometry = int(bandbal_geometry)
        if self.bandbal_geometry not in _FRAMES:
            raise ValueError(f"bandbal_geometry must be 256 or 512, got {bandbal_geometry!r}")
        if self.bandbal_geometry == 512 and bandbal_target != "teacher":
            raise ValueError("bandbal_geometry 512 compares with the 512 TEACHER; the GT input is only 256")
        frame = _FRAMES[self.bandbal_geometry]["names"]
        names = list(bandbal_levels) if bandbal_levels is not None else list(frame)
        bad = [n for n in names if n not in frame]
        if bad or not names:
            raise ValueError(f"bandbal_levels {names} must be a non-empty subset of {frame}")
        self.bb_names = tuple(n for n in frame if n in names)             # frame order, coarse -> fine
        # plain attributes (never buffers): the state_dict key set stays identical to every other arm
        full = build_level_masks("cpu", side=self.bandbal_geometry)
        self._bb_masks = full[[frame.index(n) for n in self.bb_names]]
        if bandbal_energies is not None:
            e = torch.tensor([float(v) for v in bandbal_energies], dtype=torch.float32)
            if e.numel() != len(self.bb_names):
                raise ValueError(f"bandbal_energies needs {len(self.bb_names)} values ({self.bb_names})")
            self._bb_E, self._bb_frozen = e, True
        else:
            self._bb_E, self._bb_frozen, self._bb_sum, self._bb_n = None, False, None, 0
        print(f"[bandbal] target={bandbal_target} geometry={self.bandbal_geometry} levels={','.join(self.bb_names)} "
              f"weight={self.bandbal_weight} ({bandbal_normalize}); energies "
              + ("given" if self._bb_frozen else f"measured on the first {self.bandbal_energy_batches} batches"))

    # ------------------------------------------------------------------ helpers
    def _target(self, inputs):
        if self.bandbal_target == "gt":
            return inputs
        t = self.teacher_input
        if t is None:
            raise RuntimeError("bandbal_target 'teacher' but no teacher was stashed (use AutoencoderKLSteerableEMA "
                               "or a subclass, with a FileListSRTeacher* dataset)")
        if t.shape[0] != inputs.shape[0]:
            raise RuntimeError(f"teacher batch {tuple(t.shape)} does not match inputs {tuple(inputs.shape)}")
        if self.bandbal_geometry == 256 and tuple(t.shape[-2:]) != tuple(inputs.shape[-2:]):
            t = F.interpolate(t.float(), size=inputs.shape[-2:], mode="area")
        if self.bandbal_geometry == 512 and tuple(t.shape[-2:]) != (512, 512):
            raise RuntimeError(f"bandbal_geometry 512 needs a 512 teacher, got {tuple(t.shape)}")
        return t.float().clamp(-1, 1)

    def _energies(self, lt, training):
        side = int(lt.shape[-1]); b = _geom(side)[2]
        m = slice(b, side - b)
        if not self._bb_frozen and training:
            with torch.no_grad():
                e = (lt[..., m, m] ** 2).mean(dim=(1, 2, 3, 4)).detach()
                self._bb_sum = e if self._bb_sum is None else self._bb_sum + e
                self._bb_n += 1
                if self._bb_n >= self.bandbal_energy_batches:
                    E = self._bb_sum / self._bb_n
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(E)
                        E = E / torch.distributed.get_world_size()
                    self._bb_E, self._bb_frozen = E.detach().float(), True
                    print("[bandbal] level energies frozen (" + ", ".join(self.bb_names) + "): "
                          + " ".join(f"{v:.3e}" for v in self._bb_E.tolist()))
                    return self._bb_E
                return (self._bb_sum / self._bb_n).float()
        if self._bb_E is None:          # validation before any training batch: measure on this batch, not stored
            with torch.no_grad():
                return (lt[..., m, m] ** 2).mean(dim=(1, 2, 3, 4)).detach().float()
        return self._bb_E

    def bandbal(self, reconstructions, inputs, training):
        masks = self._bb_masks
        if masks.device != reconstructions.device:
            masks = self._bb_masks = masks.to(reconstructions.device)
        if self._bb_E is not None and self._bb_E.device != reconstructions.device:
            self._bb_E = self._bb_E.to(reconstructions.device)
        with torch.autocast(device_type=reconstructions.device.type, enabled=False):   # FFTs in fp32
            pred = reconstructions.float()
            if self.bandbal_geometry == 512:      # student 256 -> 512 through the steerable term's own upsampler
                pred = self._up(pred, (512, 512))
            side = int(pred.shape[-1]); b = _geom(side)[2]; m = slice(b, side - b)
            lp = level_filter(pred, masks)
            with torch.no_grad():
                lt = level_filter(self._target(inputs), masks)
            E = self._energies(lt, training)
            per = ((lp - lt)[..., m, m] ** 2).mean(dim=(1, 2, 3, 4)) / E.clamp_min(1e-12)
        return per.mean(), per.detach()

    # ------------------------------------------------------------------ forward
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx, global_step,
                last_layer=None, cond=None, split="train", weights=None):
        loss, log = super().forward(inputs, reconstructions, posteriors, optimizer_idx, global_step,
                                    last_layer=last_layer, cond=cond, split=split, weights=weights)
        if optimizer_idx != 0 or self.bandbal_weight == 0.0:
            return loss, log
        bb, per = self.bandbal(reconstructions, inputs, training=(split == "train"))
        scale = float(inputs[0].numel()) if self.bandbal_normalize == "nll_units" else 1.0
        term = self.bandbal_weight * scale * bb
        loss = loss + term
        log[f"{split}/bandbal"] = bb.detach()                  # unit-free, comparable with the probe
        log[f"{split}/bandbal_term"] = term.detach()           # loss units
        log[f"{split}/bandbal_share"] = (term / loss).detach()
        for name, v in zip(self.bb_names, per):
            log[f"{split}/bandbal_{name.replace('*', 's')}"] = v
        log[f"{split}/total_loss"] = loss.clone().detach().mean()
        return loss, log
