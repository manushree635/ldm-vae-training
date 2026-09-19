from __future__ import annotations
"""Learned 2x sub-pixel upsampler (conv + PixelShuffle) that starts as bicubic.

Used by the steerable loss (losses_steerable.py, `steerable_upsampler: subpixel`)
in place of the fixed `F.interpolate(mode="bicubic")` that lifts the 256 student
to the 512 teacher grid. It is LOSS-SIDE ONLY: the decoder still emits 256, and
the head is never used at inference.

WHAT IT IS. One 5x5 kernel per output phase (2x2 = 4 phases), shared across the
image channels, applied with replicate padding and interleaved by
`F.pixel_shuffle(., 2)`. Linear, no bias, no nonlinearity: a learnable polyphase
interpolation filter, 100 parameters. Bicubic 2x IS a polyphase filter with 4 taps
per phase, so it is exactly representable here, and the head is initialised to it.

WHY REPLICATE PADDING. `F.interpolate` clamps source indices at the border, which
is replicate padding. Measured against F.interpolate(bicubic, align_corners=False)
on random [2,3,256,256]: replicate pad -> max abs diff 9.5e-7 (fp32 rounding);
zero pad -> 1.16 at the borders. So `steerable_upsampler: subpixel` is bit-for-bit
the bicubic arm at step 0, borders included.

WHY ONE SHARED KERNEL. Bicubic is channel-independent, and the steerable term is
computed on luminance, so per-channel kernels would just be three copies of the
same 100 numbers with 1/3 of the gradient each.

WHY DC-NORMALISE (default on). Each phase kernel is divided by its own sum in
forward, so every output pixel remains a weighted average with weights summing to
1. Without it the head could satisfy a chunk of the term with a global gain or
offset that has nothing to do with what the decoder produces. A no-op at init:
bicubic taps already sum to 1.

AUTOCAST. `upsample_bicubic2d` runs in fp32 under autocast; a conv would run in
fp16. The forward disables autocast and works in float32 so the fp16cache configs
see the same numbers through either path.
"""

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cubic(t: float, a: float = -0.75) -> float:
    """Keys cubic convolution kernel, PyTorch's A=-0.75 flavour."""
    t = abs(float(t))
    if t <= 1.0:
        return (a + 2.0) * t ** 3 - (a + 3.0) * t ** 2 + 1.0
    if t < 2.0:
        return a * t ** 3 - 5.0 * a * t ** 2 + 8.0 * a * t - 4.0 * a
    return 0.0


def bicubic_2x_taps_1d() -> torch.Tensor:
    """[2, 5] float64: row p = 1D bicubic taps for output phase p over input offsets -2..+2.

    align_corners=False maps output pixel o to input coordinate (o + 0.5)/2 - 0.5.
    With o = 2i + p that is i + (p/2 - 0.25): phase 0 sits a quarter pixel LEFT of
    input i and uses offsets -2..+1; phase 1 sits a quarter pixel RIGHT and uses
    -1..+2. Both fit a 5-wide window centred on i, with one zero tap each.
    """
    taps = torch.zeros(2, 5, dtype=torch.float64)
    for p in range(2):
        src = p / 2.0 - 0.25
        for j, off in enumerate(range(-2, 3)):
            taps[p, j] = _cubic(off - src)
    return taps


def bicubic_2x_kernels() -> torch.Tensor:
    """[4, 5, 5] float32 separable bicubic kernels, phase index = py*2 + px.

    That ordering is the one `F.pixel_shuffle(., 2)` expects: channel
    c*4 + py*2 + px lands at output (2h + py, 2w + px).
    """
    t = bicubic_2x_taps_1d()
    ks = [torch.outer(t[py], t[px]) for py in range(2) for px in range(2)]
    return torch.stack(ks, dim=0).float()


class SubPixelUp2x(nn.Module):
    """2x upsample = replicate-pad -> grouped 5x5 conv (one kernel set, all channels) -> PixelShuffle(2).

    `weight` [4, 1, 5, 5] is the only parameter. `weight_init` is a persistent
    buffer holding the bicubic init so `drift()` stays meaningful across a resume.
    """

    def __init__(self, normalize_dc: bool = True):
        super().__init__()
        self.normalize_dc = bool(normalize_dc)
        k = bicubic_2x_kernels().unsqueeze(1)                 # [4, 1, 5, 5]
        self.weight = nn.Parameter(k.clone())
        self.register_buffer("weight_init", k.clone())

    def effective_weight(self) -> torch.Tensor:
        w = self.weight
        if self.normalize_dc:
            w = w / w.sum(dim=(-2, -1), keepdim=True)
        return w

    def drift(self) -> torch.Tensor:
        """max |weight - bicubic init|. 0 exactly until the optimiser moves the head."""
        return (self.weight.detach() - self.weight_init).abs().max()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("SubPixelUp2x expects [N,C,H,W], got {}".format(tuple(x.shape)))
        c = int(x.shape[1])
        amp_off = (torch.autocast(device_type="cuda", enabled=False)
                   if x.is_cuda else contextlib.nullcontext())
        with amp_off:
            xf = x.float()
            k = self.effective_weight()                       # [4, 1, 5, 5]
            k = k.unsqueeze(0).expand(c, 4, 1, 5, 5).reshape(c * 4, 1, 5, 5)
            xp = F.pad(xf, (2, 2, 2, 2), mode="replicate")
            y = F.conv2d(xp, k, groups=c)                     # [N, 4C, H, W]
            return F.pixel_shuffle(y, 2)                      # [N, C, 2H, 2W]

    def extra_repr(self) -> str:
        return "normalize_dc={}, params={}".format(self.normalize_dc, self.weight.numel())
