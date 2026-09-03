from __future__ import annotations
"""Progressive band-weight schedule for the steerable term.

Ported from distillation_nerc/lightning_module.py:_steerable_scale_level_groups
and _resolve_steerable_band_weights. The unit of scheduling is a SCALE LEVEL, not
an individual band: level li is the four oriented bands ('band', li, 0..3), with
residual_highpass folded into level 0 (finest) and residual_lowpass into level
height-1 (coarsest). So a height-5 pyramid has 5 schedulable levels covering all
22 band keys.

Inactive levels get weight 0. Because the aggregation in steerable_rel_l1 is a
WEIGHTED MEAN -- sum(w*rel)/sum(w) -- zeroing a level automatically renormalises
the remaining ones, so the term does not shrink just because fewer levels are on.
That is why `normalize_by_active_levels` is NOT applied here by default: nerc
carries an extra 1/active factor on top of the same weighted mean, which scales
the term DOWN as levels activate. Exposed as an option, off unless asked for.
"""


def scale_level_groups(height: int, nbands: int, order: str = "coarse_to_fine"):
    """Ordered band-key groups, one per scale level. groups[0] activates first."""
    height = max(1, int(height))
    nbands = max(1, int(nbands))
    fine_to_coarse = []
    for li in range(height):
        keys = [("band", li, bi) for bi in range(nbands)]
        if li == 0:
            keys = [("residual_highpass",)] + keys
        if li == height - 1:
            keys = keys + [("residual_lowpass",)]
        fine_to_coarse.append(keys)
    if str(order).strip().lower() == "fine_to_coarse":
        return fine_to_coarse
    return list(reversed(fine_to_coarse))


def parse_schedule(schedule, default_ramp_epochs: int = 0):
    """[{epoch, active_levels|active_bands, ramp_epochs?}, ...] -> sorted tuples."""
    if not schedule:
        return []
    entries = []
    for item in schedule:
        if isinstance(item, dict):
            epoch = item.get("epoch")
            lv = item.get("active_levels", item.get("active_bands", item.get("num_bands")))
            ramp = item.get("ramp_epochs", default_ramp_epochs)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            epoch, lv = item[0], item[1]
            ramp = item[2] if len(item) > 2 else default_ramp_epochs
        else:
            raise ValueError("unparseable band-schedule entry: {!r}".format(item))
        if epoch is None or lv is None:
            raise ValueError("band-schedule entry needs epoch and active_levels: {!r}".format(item))
        entries.append((int(epoch), int(lv), int(ramp)))
    return sorted(entries, key=lambda t: t[0])


def active_levels_at(epoch: int, schedule, num_levels: int) -> int:
    entries = parse_schedule(schedule)
    active = 0
    for e, lv, _ in entries:
        if epoch >= e:
            active = lv
    return max(0, min(int(num_levels), int(active)))


def resolve_band_weights(epoch, base_weights, *, height, nbands, order,
                         schedule, ramp_epochs=0, normalize_by_active_levels=False):
    """Per-epoch {str(band_key): weight}. Returns base_weights unchanged if no schedule."""
    entries = parse_schedule(schedule, default_ramp_epochs=int(ramp_epochs))
    if not entries:
        return base_weights

    groups = scale_level_groups(height, nbands, order)
    num_levels = len(groups)

    def base_for(key):
        if not isinstance(base_weights, dict):
            return 1.0
        for cand in (key, str(key)):
            if cand in base_weights:
                return float(base_weights[cand])
        for dk, dv in base_weights.items():
            if str(dk) == str(key):
                return float(dv)
        raise KeyError("no base weight for band {!r}".format(key))

    active = 0
    stage_epoch = None
    stage_ramp = int(ramp_epochs)
    for e, lv, r in entries:
        if epoch >= e:
            active, stage_epoch, stage_ramp = int(lv), int(e), int(r)
    active = max(0, min(num_levels, active))

    scale = [0.0] * num_levels
    if stage_epoch is not None:
        prev = 0
        for e, lv, _ in entries:
            if e < stage_epoch:
                prev = int(lv)
            else:
                break
        prev = max(0, min(num_levels, prev))
        # A ramp needs something to fade in AGAINST. With prev == 0 -- the first
        # stage of any schedule -- there are no already-on levels, so the fraction
        # multiplies every active weight uniformly, and a uniform factor cancels
        # in the weighted mean sum(w*rel)/sum(w) that steerable_rel_l1 computes.
        # Measured on the f2c run: wsum 0.29 -> 0.58 across the stage-1 ramp moved
        # rel_l1 by 0.0006, i.e. nothing. Applying it anyway leaves the objective
        # untouched but logs steerable_wsum at half its true value, which reads as
        # "the term is being eased in" when it is in fact at full strength -- and
        # at epoch 0 rel_l1 is the HIGHEST of the run. Worse, raising ramp_epochs
        # to soften the start would move wsum and change nothing, with no error.
        # So skip the ramp when there is nothing to ramp against: a flat first
        # stage now reads flat, and a ramp_epochs that does nothing shows nothing.
        ramping = stage_ramp > 0 and prev > 0
        for gi in range(0, prev):
            scale[gi] = 1.0
        for gi in range(prev, active):
            if ramping:
                scale[gi] = min(1.0, max(0.0, (epoch - stage_epoch + 1) / float(stage_ramp)))
            else:
                scale[gi] = 1.0

    if normalize_by_active_levels and active > 0:
        scale = [s / float(active) for s in scale]

    out = {}
    for gi, keys in enumerate(groups):
        for k in keys:
            out[str(k)] = base_for(k) * scale[gi]

    if not any(v > 0.0 for v in out.values()):
        raise ValueError(
            "band schedule leaves every band at weight 0 at epoch {} -- the weighted "
            "mean in steerable_rel_l1 would divide by zero. Give the first stage "
            "epoch 0.".format(epoch))
    return out
