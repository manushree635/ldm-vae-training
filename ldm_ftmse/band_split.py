from __future__ import annotations
"""Fixed coarse/fine split of an image at a steerable-pyramid level boundary.

WHAT THIS IS FOR. The dual-branch decoder (autoencoder_dualbranch.py) has two output
tails and combines them as

    out = LP(b1) + HP(b2) = b2 + LP(b1 - b2)

where LP keeps the pyramid's COARSE levels (>= split_level, plus residual_lowpass) and
HP = 1 - LP keeps the fine ones (residual_highpass + levels 0 .. split_level-1). With
the default split_level=3 and the 512 loss pyramid, branch 2 owns residual_highpass and
levels 0,1,2 -- the ~16-256 cycles/image octaves -- and branch 1 owns the rest.

WHY AN FFT MASK RATHER THAN A PYRAMID SWAP. Building both pyramids, swapping levels and
calling reconstruct() is the literal operation, but SCFpyr_PyTorch.reconstruct rebuilds
its masks in NumPy on EVERY call (no cache), which is far too slow for a training step.
The complex steerable pyramid is linear, shift-invariant and perfect-reconstruction, so
"reconstruct from coarse levels only" IS a fixed radial filter: one H, computed once per
size, then applied with two FFTs. scripts/test_dual_branch.py pins this equivalence
against the real level swap.

H is obtained by pushing an impulse through a pyramid TRUNCATED to `split_level` oriented
levels and keeping only its lowpass residual. Under perfect reconstruction the squared
masks sum to 1 at every frequency, so that truncated lowpass is exactly the complement of
(residual_highpass + levels 0..split_level-1) of the taller pyramid the loss builds.

RESOLUTION. The loss builds its pyramid on the TEACHER grid (`match_mode: upsample`,
student 256 -> 512), so band indices are relative to a grid `pyr_scale` times the decoder
output. H is therefore computed at (pyr_scale*h, pyr_scale*w) and the central h x w
frequencies are cropped out. The cutoff then sits at a fixed cycles/PIXEL of the output,
so decoding at 512 at eval time splits at the same physical frequency it was trained at.

BORDERS. The pyramid -- and any FFT filter -- is circular. `mix` and `lowpass` reflect-pad
before the transform and crop after, so the wrap never lands on a non-periodic edge. This
matters in `replace` mode, where nothing constrains b2's coarse content and b1 - b2 can be
large at low frequencies.
"""

import torch

from ._steerable_pyr.SCFpyr_PyTorch import SCFpyr_PyTorch

#: (h, w, split_level, order, pyr_scale, device, dtype) -> H
_H_CACHE = {}

#: Frequency content the crop throws away must be numerically zero: the coarse filter is
#: ~0 an octave below the output grid's Nyquist, so anything above it is rounding error.
_CROP_TOL = 1e-5


def _pyr_for(split_level: int, order: int, device):
    # height counts the two residuals, as in steerable_rel_l1. `split_level` oriented
    # levels + highpass + lowpass = split_level + 2.
    return SCFpyr_PyTorch(height=int(split_level) + 2, nbands=int(order) + 1,
                          scale_factor=2, device=device)


def coarse_transfer(h: int, w: int, split_level: int, *, order: int = 3, pyr_scale: int = 2,
                    device=None, dtype=torch.float32) -> torch.Tensor:
    """Real transfer function H (shape [h, w], un-shifted FFT layout) of the COARSE part.

    H is 1 at DC, 0 above the split, and H + (1 - H) = 1 by construction, so the coarse and
    fine halves partition the image exactly.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    key = (int(h), int(w), int(split_level), int(order), int(pyr_scale), str(device), dtype)
    cached = _H_CACHE.get(key)
    if cached is not None:
        return cached

    H2, W2 = int(pyr_scale) * int(h), int(pyr_scale) * int(w)
    pyr = _pyr_for(split_level, order, device)

    impulse = torch.zeros(1, 1, H2, W2, device=device, dtype=torch.float32)
    impulse[0, 0, 0, 0] = 1.0
    coeff = pyr.build(impulse)

    # Keep ONLY the lowpass residual: zero the highpass and every oriented band. The
    # structure has to survive intact -- reconstruct() checks len(coeff[1]) == nbands.
    coeff[0] = torch.zeros_like(coeff[0])
    for lvl in range(1, len(coeff) - 1):
        coeff[lvl] = [torch.zeros_like(b) for b in coeff[lvl]]
    resp = pyr.reconstruct(coeff)                      # [1, H2, W2], real

    big = torch.fft.fft2(resp)[0].real                 # zero-phase filter -> real transfer
    big = torch.fft.fftshift(big)

    # Crop the central h x w frequencies: cycles-per-image is preserved, so this is the
    # same physical cutoff on the smaller grid.
    r0, c0 = (H2 - int(h)) // 2, (W2 - int(w)) // 2
    small = big[r0:r0 + int(h), c0:c0 + int(w)]
    dropped = big.abs().sum() - small.abs().sum()
    if float(dropped) > _CROP_TOL * float(big.abs().sum().clamp_min(1e-12)):
        raise RuntimeError(
            "coarse_transfer: the {}x{} crop drops {:.3g} of transfer mass -- the split at "
            "level {} is not well below the output grid's Nyquist".format(h, w, float(dropped),
                                                                          split_level))
    H = torch.fft.ifftshift(small).to(dtype).contiguous()
    _H_CACHE[key] = H
    return H


def _filter_padded(x: torch.Tensor, split_level: int, pad: int, *, order: int, pyr_scale: int):
    """H * x, reflect-padded so the circular FFT never meets a non-periodic edge."""
    pad = int(pad)
    if pad > 0:
        x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    H = coarse_transfer(x.shape[-2], x.shape[-1], split_level, order=order,
                        pyr_scale=pyr_scale, device=x.device, dtype=x.dtype)
    y = torch.fft.ifft2(torch.fft.fft2(x) * H).real
    if pad > 0:
        y = y[..., pad:y.shape[-2] - pad, pad:y.shape[-1] - pad]
    return y


def _fp32(fn):
    """The pyramid and these FFTs are fp32-only; keep them out of any autocast region."""
    def wrapper(*args, **kwargs):
        first = args[0]
        if first.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                return fn(*[a.float() if torch.is_tensor(a) else a for a in args], **kwargs)
        return fn(*[a.float() if torch.is_tensor(a) else a for a in args], **kwargs)
    return wrapper


@_fp32
def lowpass(x: torch.Tensor, split_level: int, pad: int = 32, *, order: int = 3, pyr_scale: int = 2):
    """LP(x): the pyramid's coarse half (levels >= split_level + residual_lowpass)."""
    return _filter_padded(x, split_level, pad, order=order, pyr_scale=pyr_scale)


@_fp32
def mix(b1: torch.Tensor, b2: torch.Tensor, split_level: int, pad: int = 32, *,
        order: int = 3, pyr_scale: int = 2):
    """LP(b1) + HP(b2), written as b2 + LP(b1 - b2) so b1 == b2 returns b1 EXACTLY."""
    if b1.shape != b2.shape:
        raise ValueError("dual-branch outputs disagree: {} vs {}".format(tuple(b1.shape), tuple(b2.shape)))
    return b2 + _filter_padded(b1 - b2, split_level, pad, order=order, pyr_scale=pyr_scale)
