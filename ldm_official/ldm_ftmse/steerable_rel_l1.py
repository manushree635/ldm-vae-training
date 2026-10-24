from __future__ import annotations
"""Steerable-pyramid relative-L1, ported from distillation_nerc/losses.py.

`from __future__ import annotations` above is load-bearing: the source uses
`list[...]` and `X | None` in signatures, which Python evaluates at def time and
which do not exist before 3.9/3.10. This runs on 3.8.

Four deliberate departures from the source, each fixing something rather than
transcribing it:

  1. Teacher and student are built as TWO pyramid calls, with the teacher under
     no_grad. Verified bit-identical to the source's single concatenated build
     (max abs diff 0), and it both makes the "teacher branch is not detached"
     hazard structurally impossible and halves the retained autograd graph.
  2. `_to_luminance_t01` no longer guesses the input range from
     `x.min().item() < 0`. Both tensors here are [-1,1] by construction, the
     guess costs a device->host sync per call, and it is wrong for an
     all-nonnegative image (measured: 2/300 real crops would misfire alone).
  3. Band-weight resolution is strict -- an unmatched band or an unconsumed dict
     key raises. The source silently assigns 1.0, which hides both a typo and
     the 18-vs-22-band resolution trap.
  4. The pyramid import is a hard failure, not `except: SCFpyr_PyTorch = None`.
     A silent None there means training runs with the term quietly absent.
"""

import math

import torch
import torch.nn.functional as F

from ._steerable_pyr.SCFpyr_PyTorch import SCFpyr_PyTorch

# One pyramid object per (height, nbands, scale_factor, device). The object caches its
# frequency-domain masks, which are built in NumPy on CPU and dominate cost if rebuilt.
_STEERABLE_PYR_CACHE: dict = {}


def _max_steerable_pyr_height_for_size(spatial_size: int) -> int:
    """Max SCFpyr_PyTorch height for a square side length (mirrors build()'s guard)."""
    side = max(1, int(spatial_size))
    return max(1, int(math.floor(math.log2(side)) - 2))


def _get_steerable_pyr(*, height: int, nbands: int, scale_factor: int, device):
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda:0")
    key = (int(height), int(nbands), int(scale_factor), str(dev))
    pyr = _STEERABLE_PYR_CACHE.get(key)
    if pyr is None:
        pyr = SCFpyr_PyTorch(height=int(height), nbands=int(nbands),
                             scale_factor=int(scale_factor), device=dev)
        _STEERABLE_PYR_CACHE[key] = pyr
    return pyr


def _to_luminance_t01(img_nchw: torch.Tensor) -> torch.Tensor:
    """RGB [-1,1] -> luminance Y in [0,1]. Input [N,3,H,W], output [N,1,H,W]."""
    x = (img_nchw * 0.5 + 0.5).clamp(0.0, 1.0)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _to_channel_t01(img_nchw: torch.Tensor, channel: int) -> torch.Tensor:
    """One RGB channel of a [-1,1] image, in [0,1]. Input [N,3,H,W] -> [N,1,H,W]."""
    x = (img_nchw * 0.5 + 0.5).clamp(0.0, 1.0)
    return x[:, channel : channel + 1]


def _safe_complex_abs(complex_tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """|z| via sqrt(re^2 + im^2 + eps).

    Deliberately not torch.abs() on a complex tensor: this form keeps the backward
    well-behaved on older torch, and the eps keeps the gradient finite at zero.
    """
    return torch.sqrt(complex_tensor.real ** 2 + complex_tensor.imag ** 2 + eps)


def _box_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    """F.avg_pool2d(x, k, stride=1, padding=k//2) over the last two dims, any rank.
    Defaults as torch's (count_include_pad=True), so borders see zero padding."""
    shp = x.shape
    y = F.avg_pool2d(x.reshape(-1, 1, shp[-2], shp[-1]), int(k), stride=1, padding=int(k) // 2)
    return y.reshape(shp)


def _iter_steerable_bands(coeff_batch):
    # coeff[0]      residual highpass, real    [N,H,W]
    # coeff[1:-1]   levels of complex bands    [N,h,w], level 0 = FINEST
    # coeff[-1]     residual lowpass, real     [N,h,w]
    yield ("residual_highpass",), coeff_batch[0]
    for li, level in enumerate(coeff_batch[1:-1]):
        for bi, band in enumerate(level):
            yield ("band", int(li), int(bi)), band
    yield ("residual_lowpass",), coeff_batch[-1]


def _sorted_band_keys(keys):
    return sorted(list(keys), key=lambda k: str(k))


def steerable_band_key_to_log_name(key) -> str:
    if key == ("residual_highpass",):
        return "residual_highpass"
    if key == ("residual_lowpass",):
        return "residual_lowpass"
    if isinstance(key, tuple) and len(key) == 3 and key[0] == "band":
        return "band_{}_{}".format(int(key[1]), int(key[2]))
    return str(key).strip("()").replace(", ", "_").replace("'", "")


def _resolve_band_weights(keys, band_weights, *, device=None, dtype=None, strict=True):
    """Positional weight tensor aligned to `keys`.

    Accepts None (all 1.0), a list/tuple of len(keys), or a dict keyed by the band
    tuple or by `str(tuple)` -- the JSON/YAML form, e.g. "('band', 0, 0)".

    strict=True (default, and a departure from the source) raises when a band gets
    no explicit weight or when a supplied key is never consumed. The source falls
    back to 1.0 in silence, which hides a mistyped key AND the resolution trap: at
    256px the pyramid has 18 bands, not 22, so the four ('band', 4, *) entries --
    which carry the HEAVIEST weight in the diffweight ramp -- would vanish unnoticed
    and the objective would quietly change meaning.
    """
    n = len(keys)
    if band_weights is None:
        return torch.tensor([1.0] * n, device=device, dtype=dtype)

    if isinstance(band_weights, (list, tuple)):
        if len(band_weights) != n:
            raise ValueError(
                "steerable band_weights length {} != num bands {}".format(len(band_weights), n))
        return torch.tensor([float(x) for x in band_weights], device=device, dtype=dtype)

    if not isinstance(band_weights, dict):
        # OmegaConf hands us a DictConfig, which is NOT a dict.
        if hasattr(band_weights, "keys"):
            band_weights = {k: band_weights[k] for k in band_weights.keys()}
        else:
            raise TypeError("steerable band_weights must be None, list/tuple, or dict-like")

    by_str = {str(k): float(v) for k, v in band_weights.items()}
    w, unmatched = [], []
    for k in keys:
        sk = str(k)
        if sk in by_str:
            w.append(by_str.pop(sk))
        else:
            unmatched.append(sk)
            w.append(1.0)

    if strict and (unmatched or by_str):
        raise ValueError(
            "steerable band_weights did not resolve cleanly for {} bands.\n"
            "  bands with no weight given ({}): {}\n"
            "  weights given but never used ({}): {}\n"
            "Keys must be repr(tuple), e.g. \"('band', 0, 0)\" and "
            "\"('residual_highpass',)\" -- note the spaces and the trailing comma. "
            "Unused ('band', 4, *) keys usually mean the pyramid ran at 256 (18 bands) "
            "instead of 512 (22 bands)."
            .format(n, len(unmatched), unmatched, len(by_str), sorted(by_str))
        )
    return torch.tensor(w, device=device, dtype=dtype)


def _finest_band_hf_mask(channel_pairs, gamma: float, w_max, eps: float) -> torch.Tensor:
    """Content-adaptive mask on the finest band, built ONLY from the teacher so it
    cannot reinforce student artifacts. E = sqrt(sum |P_T|^2 + eps), normalised to
    unit mean per image. Detached throughout."""
    mags_sq = []
    for bt_map, _ in channel_pairs:
        for k, band in bt_map.items():
            if isinstance(k, tuple) and len(k) == 3 and k[0] == "band" and int(k[1]) == 0:
                mag = _safe_complex_abs(band, eps=float(eps)) if torch.is_complex(band) else band.abs()
                mags_sq.append(mag * mag)
    energy = torch.sqrt(torch.stack(mags_sq, dim=0).sum(dim=0) + float(eps)).detach()
    e_gamma = energy.clamp_min(0.0) ** float(gamma)
    mask = e_gamma / (e_gamma.mean(dim=(-2, -1), keepdim=True) + float(eps))
    if w_max is not None:
        mask = mask.clamp(max=float(w_max))
        mask = mask / (mask.mean(dim=(-2, -1), keepdim=True) + float(eps))
    return mask.detach()


def _rel_l1_from_coeff_pairs(
    coeffs_t,
    coeffs_s,
    n: int,
    *,
    band_weights,
    pyr_complex: bool,
    complex_metric: str,
    eps: float,
    teacher_offsets=None,
    student_offsets=None,
    hf_mask_gamma=None,
    hf_mask_w_max=None,
    rel_alpha: float = 1.0,
    strict_band_weights: bool = True,
    mag_pool_kernel: int = 0,
    complex_metric_overrides=None,
):
    """Per-band relative L1 between two already-built pyramids.

        per_band = mean|P_T - P_S| / (mean|P_T|) ** rel_alpha

    rel_alpha=1 (default) makes every band report a FRACTION of its own energy, so a
    weak fine band costs as much proportionally as a strong coarse one -- this is the
    mechanism that makes the term preserve texture where plain L1/MSE does not.

    Reduction is a weighted MEAN (divided by sum(w)), so scaling every weight by a
    constant is a no-op; only ratios matter.
    """
    keys = _sorted_band_keys(k for k, _ in _iter_steerable_bands(coeffs_t))
    sample = next(v for _, v in _iter_steerable_bands(coeffs_t))
    w = _resolve_band_weights(keys, band_weights, device=sample.device,
                              dtype=torch.float32, strict=strict_band_weights)
    wsum = w.sum().clamp_min(float(eps))
    # complex_metric_overrides: {str(band_key): "magnitude" | "complex"} -- a per-band
    # metric, e.g. magnitude on ('band', 0, *) where phase is unrecoverable and complex
    # elsewhere. Keys are matched like band_weights (str of the tuple) and strictly:
    # an unmatched key or an unknown metric raises. None/{} = the single global metric.
    metric_by_key = {}
    if complex_metric_overrides:
        ov = {str(k): str(v).strip().lower() for k, v in dict(complex_metric_overrides).items()}
        known = {str(k) for k in keys}
        bad = sorted(set(ov) - known)
        if bad:
            raise ValueError("complex_metric_overrides has keys that match no band: {} (bands: {})".format(bad, sorted(known)))
        for v in ov.values():
            if v not in ("magnitude", "complex"):
                raise ValueError("complex_metric_overrides values must be 'magnitude' or 'complex', got {!r}".format(v))
        metric_by_key = ov

    channel_pairs = []
    if teacher_offsets is None or student_offsets is None:
        bt = {k: v[:n] for k, v in _iter_steerable_bands(coeffs_t)}
        bs = {k: v[:n] for k, v in _iter_steerable_bands(coeffs_s)}
        channel_pairs.append((bt, bs))
    else:
        if len(teacher_offsets) != len(student_offsets):
            raise ValueError("teacher_offsets and student_offsets must have the same length")
        for off_t, off_s in zip(teacher_offsets, student_offsets):
            bt = {k: v[off_t: off_t + n] for k, v in _iter_steerable_bands(coeffs_t)}
            bs = {k: v[off_s: off_s + n] for k, v in _iter_steerable_bands(coeffs_s)}
            channel_pairs.append((bt, bs))

    hf_mask = None
    if hf_mask_gamma is not None:
        hf_mask = _finest_band_hf_mask(channel_pairs, float(hf_mask_gamma), hf_mask_w_max, float(eps))

    per_channel_rel = []
    for bt_map, bs_map in channel_pairs:
        per_band_rel = []
        for k in keys:
            bt, bs_ = bt_map[k], bs_map[k]
            is_finest = isinstance(k, tuple) and len(k) == 3 and k[0] == "band" and int(k[1]) == 0
            spatial_w = hf_mask if (hf_mask is not None and is_finest) else None

            metric_k = metric_by_key.get(str(k), str(complex_metric))
            if torch.is_complex(bt):
                if bool(pyr_complex):
                    if metric_k == "complex":
                        err_map = _safe_complex_abs(bt - bs_, eps=float(eps))
                        denom_map = _safe_complex_abs(bt, eps=float(eps))
                    else:  # "magnitude" -- phase-blind
                        bt_mag = _safe_complex_abs(bt, eps=float(eps))
                        bs_mag = _safe_complex_abs(bs_, eps=float(eps))
                        if int(mag_pool_kernel) > 1:
                            # LOCAL-ENERGY match: box-pool both magnitudes before the
                            # difference, so texture displaced by a pixel or two costs
                            # less than texture that is missing. The denominator stays
                            # the UNPOOLED teacher magnitude:
                            #   r_k = mean|pool|a_t| - pool|a_s|| / (mean|a_t| + eps)
                            kp = int(mag_pool_kernel)
                            err_map = (_box_pool(bt_mag, kp) - _box_pool(bs_mag, kp)).abs()
                        else:
                            err_map = (bt_mag - bs_mag).abs()
                        denom_map = bt_mag
                else:
                    err_map = (bt.real - bs_.real).abs()
                    denom_map = bt.real.abs()
            else:
                err_map = (bt - bs_).abs()
                denom_map = bt.abs()

            if spatial_w is not None:
                per_img_l1 = (err_map * spatial_w).mean(dim=(-2, -1))
                per_img_denom = (denom_map * spatial_w).mean(dim=(-2, -1)) + float(eps)
            else:
                per_img_l1 = err_map.mean(dim=(-2, -1))
                per_img_denom = denom_map.mean(dim=(-2, -1)) + float(eps)

            if float(rel_alpha) == 1.0:
                per_band_rel.append(per_img_l1 / per_img_denom)
            else:
                per_band_rel.append(per_img_l1 / per_img_denom.pow(float(rel_alpha)))
        per_channel_rel.append(torch.stack(per_band_rel, dim=0))

    rel_kn = torch.stack(per_channel_rel, dim=0).mean(dim=0)
    weighted = (rel_kn * w[:, None]).sum(dim=0) / wsum
    per_band_means = {steerable_band_key_to_log_name(k): rel_kn[i].mean()
                      for i, k in enumerate(keys)}
    return weighted.mean(), per_band_means


def compute_steerable_rel_l1(
    teacher_image: torch.Tensor,
    student_image: torch.Tensor,
    *,
    pyr_height: int,
    pyr_order: int,
    band_weights=None,
    match_mode: str = "upsample",
    down_mode: str = "bicubic",
    color_mode: str = "luminance",
    pyr_complex: bool = True,
    complex_metric: str = "magnitude",
    eps: float = 1e-8,
    hf_mask_gamma=None,
    hf_mask_w_max=None,
    rel_alpha: float = 1.0,
    strict_band_weights: bool = True,
    upsample_fn=None,
    mag_pool_kernel: int = 0,
    complex_metric_overrides=None,
):
    """Steerable-pyramid relative L1 between a teacher and a student image.

    Both are [N,3,H,W] in [-1,1]. Returns (scalar, {band_name: mean}).

    match_mode="upsample" resizes the STUDENT up to the teacher's size, so with a 256
    student and a 512 teacher the comparison happens at 512 -- which is what gives 5
    oriented levels (22 bands) and lets the diffweight ramp's heaviest ('band', 4, *)
    entries exist at all. "downsample" would compare at 256, where the pyramid caps at
    4 levels (18 bands) and those weights silently disappear.

    upsample_fn: optional callable student -> student at the teacher's size, used in
    place of the bicubic F.interpolate in "upsample" mode (e.g. a learned
    ldm_ftmse.subpixel_up.SubPixelUp2x). None keeps the bicubic path untouched.
    "downsample" ignores it: pixel shuffle cannot go down.
    """
    # mag_pool_kernel > 1 box-pools the oriented bands' MAGNITUDES before differencing
    # (magnitude metric only; residual bands keep their signed comparison). 0/1 = off,
    # which is the original code path exactly.
    if int(mag_pool_kernel) > 1:
        if str(complex_metric) != "magnitude" or not bool(pyr_complex):
            raise ValueError("mag_pool_kernel needs pyr_complex=True and complex_metric='magnitude'")
        if int(mag_pool_kernel) % 2 == 0:
            raise ValueError("mag_pool_kernel must be odd so the pooled map keeps its size")
    match = (match_mode or "upsample").strip().lower()
    if match == "upsample":
        if student_image.shape[-2:] != teacher_image.shape[-2:]:
            if upsample_fn is not None:
                student_image = upsample_fn(student_image)
                if student_image.shape[-2:] != teacher_image.shape[-2:]:
                    raise ValueError(
                        "upsample_fn returned {} but the teacher is {}".format(
                            tuple(student_image.shape), tuple(teacher_image.shape)))
            else:
                student_image = F.interpolate(student_image, size=teacher_image.shape[-2:],
                                              mode="bicubic", align_corners=False)
    elif match == "downsample":
        if student_image.shape[-2:] != teacher_image.shape[-2:]:
            dm = (down_mode or "bicubic").strip().lower()
            if dm == "area":
                # Area averaging = cv2.INTER_AREA. For the integer 2x step this loss
                # actually uses (teacher 512 -> student 256) the two agree exactly: both
                # reduce to a 2x2 box mean. Bicubic instead resamples with a 4-tap kernel
                # that overshoots at edges, so it injects ringing into the very bands the
                # steerable term is measuring. align_corners is not a valid argument here.
                teacher_image = F.interpolate(teacher_image, size=student_image.shape[-2:],
                                              mode="area")
            elif dm == "bicubic":
                teacher_image = F.interpolate(teacher_image, size=student_image.shape[-2:],
                                              mode="bicubic", align_corners=False)
            else:
                raise ValueError("down_mode must be 'area' or 'bicubic', got {!r}".format(down_mode))
    else:
        raise ValueError("match_mode must be 'upsample' or 'downsample'")

    device = student_image.device
    nbands = int(pyr_order) + 1
    min_side = min(int(teacher_image.shape[-2]), int(teacher_image.shape[-1]))
    # SCFpyr height counts the two residuals, hence +2.
    requested_h = int(pyr_height) + 2
    effective_h = min(requested_h, _max_steerable_pyr_height_for_size(min_side))
    pyr = _get_steerable_pyr(height=effective_h, nbands=nbands, scale_factor=2, device=device)

    # SCFpyr_PyTorch asserts float32; keep the pyramid out of any autocast region.
    t32, s32 = teacher_image.float(), student_image.float()
    amp_off = torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else None

    def _run():
        mode = (color_mode or "luminance").strip().lower()
        n = t32.shape[0]
        if mode == "luminance":
            with torch.no_grad():
                coeffs_t = pyr.build(_to_luminance_t01(t32))
            coeffs_s = pyr.build(_to_luminance_t01(s32))
            return _rel_l1_from_coeff_pairs(
                coeffs_t, coeffs_s, n, band_weights=band_weights,
                pyr_complex=bool(pyr_complex), complex_metric=str(complex_metric),
                eps=float(eps), hf_mask_gamma=hf_mask_gamma, hf_mask_w_max=hf_mask_w_max,
                rel_alpha=float(rel_alpha), strict_band_weights=strict_band_weights,
                mag_pool_kernel=int(mag_pool_kernel),
                complex_metric_overrides=complex_metric_overrides)
        if mode == "rgb":
            with torch.no_grad():
                coeffs_t = pyr.build(torch.cat([_to_channel_t01(t32, c) for c in range(3)], dim=0))
            coeffs_s = pyr.build(torch.cat([_to_channel_t01(s32, c) for c in range(3)], dim=0))
            offs = [c * n for c in range(3)]
            return _rel_l1_from_coeff_pairs(
                coeffs_t, coeffs_s, n, band_weights=band_weights,
                pyr_complex=bool(pyr_complex), complex_metric=str(complex_metric),
                eps=float(eps), teacher_offsets=offs, student_offsets=offs,
                hf_mask_gamma=hf_mask_gamma, hf_mask_w_max=hf_mask_w_max,
                rel_alpha=float(rel_alpha), strict_band_weights=strict_band_weights,
                mag_pool_kernel=int(mag_pool_kernel),
                complex_metric_overrides=complex_metric_overrides)
        raise ValueError("color_mode must be 'luminance' or 'rgb', got {!r}".format(color_mode))

    if amp_off is None:
        return _run()
    with amp_off:
        return _run()
