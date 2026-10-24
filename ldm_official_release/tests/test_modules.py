"""Unit tests for the ldm_ftmse building blocks. CPU, no checkpoints, no data."""
import os
import tempfile
import unittest

import _env  # noqa: F401  (sys.path)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm_ftmse import band_schedule as bs
from ldm_ftmse import band_split
from ldm_ftmse.periodic_checkpoint import PeriodicCheckpoint
from ldm_ftmse.steerable_rel_l1 import compute_steerable_rel_l1
from ldm_ftmse.subpixel_up import SubPixelUp2x

KW = dict(pyr_height=5, pyr_order=3)


def img(n, side, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, side, side, generator=g) * 2 - 1


class TestBandSchedule(unittest.TestCase):
    def test_groups_cover_all_22_bands_once(self):
        groups = bs.scale_level_groups(5, 4)
        keys = [k for g in groups for k in g]
        self.assertEqual(len(groups), 5)
        self.assertEqual(len(keys), 22)
        self.assertEqual(len(set(keys)), 22)

    def test_order(self):
        c2f = bs.scale_level_groups(5, 4, "coarse_to_fine")
        f2c = bs.scale_level_groups(5, 4, "fine_to_coarse")
        self.assertIn(("residual_lowpass",), c2f[0])
        self.assertIn(("residual_highpass",), f2c[0])
        self.assertEqual(c2f, list(reversed(f2c)))

    def test_parse_schedule_accepts_dicts_and_lists_and_sorts(self):
        got = bs.parse_schedule([[10, 3], {"epoch": 0, "active_levels": 1, "ramp_epochs": 2}])
        self.assertEqual(got, [(0, 1, 2), (10, 3, 0)])
        with self.assertRaises(ValueError):
            bs.parse_schedule([{"epoch": 0}])
        with self.assertRaises(ValueError):
            bs.parse_schedule(["nonsense"])

    def test_active_levels_is_clamped(self):
        sched = [[0, 2], [5, 99]]
        self.assertEqual(bs.active_levels_at(0, sched, 5), 2)
        self.assertEqual(bs.active_levels_at(4, sched, 5), 2)
        self.assertEqual(bs.active_levels_at(5, sched, 5), 5)

    def test_no_schedule_returns_base_weights_untouched(self):
        base = {"x": 1.0}
        self.assertIs(bs.resolve_band_weights(3, base, height=5, nbands=4, order="coarse_to_fine",
                                              schedule=None), base)

    def test_inactive_levels_are_zero(self):
        w = bs.resolve_band_weights(0, None, height=5, nbands=4, order="coarse_to_fine",
                                    schedule=[[0, 2], [5, 5]])
        self.assertEqual(w[str(("residual_lowpass",))], 1.0)      # coarsest, on
        self.assertEqual(w[str(("band", 3, 0))], 1.0)             # second coarsest, on
        self.assertEqual(w[str(("band", 0, 0))], 0.0)             # finest, off
        self.assertEqual(sum(v > 0 for v in w.values()), 9)       # 4+1 and 4

    def test_first_stage_ramp_is_skipped_later_stage_ramps(self):
        sched = [{"epoch": 0, "active_levels": 1, "ramp_epochs": 4},
                 {"epoch": 2, "active_levels": 2, "ramp_epochs": 4}]
        kw = dict(height=5, nbands=4, order="coarse_to_fine", schedule=sched)
        self.assertEqual(bs.resolve_band_weights(0, None, **kw)[str(("band", 4, 0))], 1.0)
        w = bs.resolve_band_weights(2, None, **kw)
        self.assertEqual(w[str(("band", 4, 0))], 1.0)             # already on: full
        self.assertAlmostEqual(w[str(("band", 3, 0))], 0.25)      # fading in: 1/4
        self.assertEqual(bs.resolve_band_weights(9, None, **kw)[str(("band", 3, 0))], 1.0)

    def test_all_zero_schedule_raises(self):
        with self.assertRaises(ValueError):
            bs.resolve_band_weights(0, None, height=5, nbands=4, order="coarse_to_fine",
                                    schedule=[[3, 2]])

    def test_base_weights_are_kept_and_missing_key_raises(self):
        base = {str(k): 2.0 for g in bs.scale_level_groups(5, 4) for k in g}
        w = bs.resolve_band_weights(0, base, height=5, nbands=4, order="coarse_to_fine",
                                    schedule=[[0, 5]])
        self.assertTrue(all(v == 2.0 for v in w.values()))
        del base[str(("band", 2, 1))]
        with self.assertRaises(KeyError):
            bs.resolve_band_weights(0, base, height=5, nbands=4, order="coarse_to_fine",
                                    schedule=[[0, 5]])


class TestSteerableRelL1(unittest.TestCase):
    def test_identical_images_give_zero(self):
        x = img(1, 128)
        loss, bands = compute_steerable_rel_l1(x, x.clone(), **KW)
        self.assertLess(float(loss), 1e-6)
        self.assertTrue(all(float(v) < 1e-6 for v in bands.values()))

    def test_band_count_depends_on_resolution(self):
        # The trap the strict-weights check exists for: 22 bands at 512, 18 at 256.
        _, b512 = compute_steerable_rel_l1(img(1, 512), img(1, 256, 1), **KW)
        _, b256 = compute_steerable_rel_l1(img(1, 512), img(1, 256, 1), match_mode="downsample", **KW)
        self.assertEqual(len(b512), 22)
        self.assertEqual(len(b256), 18)

    def test_strict_weights_reject_unused_and_missing_keys(self):
        t, s = img(1, 128), img(1, 64, 1)
        _, bands = compute_steerable_rel_l1(t, s, **KW)
        full = {str(k): 1.0 for g in bs.scale_level_groups(5, 4) for k in g}     # 22 keys
        self.assertLess(len(bands), 22)
        with self.assertRaises(ValueError):
            compute_steerable_rel_l1(t, s, band_weights=full, **KW)
        with self.assertRaises(ValueError):
            compute_steerable_rel_l1(t, s, band_weights={"('band', 0, 0)": 1.0}, **KW)
        compute_steerable_rel_l1(t, s, band_weights=full, strict_band_weights=False, **KW)

    def test_dictconfig_weights_equal_dict_weights(self):
        t, s = img(1, 512), img(1, 256, 1)
        full = {str(k): float(1 + i) for i, g in enumerate(bs.scale_level_groups(5, 4)) for k in g}
        a, _ = compute_steerable_rel_l1(t, s, band_weights=full, **KW)
        b, _ = compute_steerable_rel_l1(t, s, band_weights=OmegaConf.create(full), **KW)
        c, _ = compute_steerable_rel_l1(t, s, **KW)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_zero_weight_band_does_not_count(self):
        t, s = img(1, 512), img(1, 256, 1)
        keys = [k for g in bs.scale_level_groups(5, 4) for k in g]
        only = {str(k): float(k == ("residual_lowpass",)) for k in keys}
        loss, bands = compute_steerable_rel_l1(t, s, band_weights=only, **KW)
        self.assertAlmostEqual(float(loss), float(bands["residual_lowpass"]), places=6)

    def test_gradient_reaches_student_only(self):
        t = img(2, 128).requires_grad_(True)
        s = img(2, 64, 1).requires_grad_(True)
        for metric in ("magnitude", "complex"):
            s.grad = None
            loss, _ = compute_steerable_rel_l1(t, s, complex_metric=metric, **KW)
            loss.backward()
            self.assertIsNone(t.grad, metric)
            self.assertTrue(torch.isfinite(s.grad).all(), metric)
            self.assertGreater(float(s.grad.abs().sum()), 0.0, metric)

    def test_black_image_stays_finite(self):
        # An all-black teacher zeroes every denominator; eps has to hold.
        t = -torch.ones(1, 3, 128, 128)
        s = img(1, 64).requires_grad_(True)
        loss, _ = compute_steerable_rel_l1(t, s, **KW)
        loss.backward()
        self.assertTrue(torch.isfinite(s.grad).all())

    def test_rgb_mode_and_bad_arguments(self):
        t, s = img(1, 128), img(1, 64, 1)
        loss, _ = compute_steerable_rel_l1(t, s, color_mode="rgb", **KW)
        self.assertTrue(torch.isfinite(loss))
        for bad in (dict(match_mode="sideways"), dict(color_mode="cmyk"),
                    dict(mag_pool_kernel=4), dict(mag_pool_kernel=3, complex_metric="complex")):
            with self.assertRaises(ValueError, msg=str(bad)):
                compute_steerable_rel_l1(t, s, **dict(KW, **bad))

    def test_upsample_fn_is_used_and_size_checked(self):
        t, s = img(1, 128), img(1, 64, 1)
        a, _ = compute_steerable_rel_l1(t, s, **KW)
        b, _ = compute_steerable_rel_l1(t, s, upsample_fn=SubPixelUp2x(), **KW)
        self.assertAlmostEqual(float(a), float(b), places=5)
        with self.assertRaises(ValueError):
            compute_steerable_rel_l1(t, s, upsample_fn=lambda x: x, **KW)


class TestSubPixelUp(unittest.TestCase):
    def test_init_is_bicubic(self):
        x = img(2, 32)
        up = SubPixelUp2x()
        ref = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        self.assertEqual(tuple(up(x).shape), (2, 3, 64, 64))
        self.assertLess(float((up(x) - ref).abs().max()), 1e-5)
        self.assertEqual(float(up.drift()), 0.0)

    def test_dc_is_preserved_after_the_weights_move(self):
        up = SubPixelUp2x(normalize_dc=True)
        with torch.no_grad():
            up.weight.add_(0.05 * torch.randn_like(up.weight))
        self.assertGreater(float(up.drift()), 0.0)
        y = up(torch.full((1, 3, 16, 16), 0.37))
        self.assertLess(float((y - 0.37).abs().max()), 1e-5)

    def test_state_dict_roundtrip_and_shape_guard(self):
        up = SubPixelUp2x()
        SubPixelUp2x().load_state_dict(up.state_dict())
        self.assertEqual(sum(p.numel() for p in up.parameters()), 100)
        with self.assertRaises(ValueError):
            up(torch.zeros(3, 16, 16))


class TestBandSplit(unittest.TestCase):
    def test_transfer_is_one_at_dc_and_bounded(self):
        H = band_split.coarse_transfer(128, 128, 2)
        self.assertAlmostEqual(float(H[0, 0]), 1.0, places=4)
        self.assertGreaterEqual(float(H.min()), -1e-4)
        self.assertLessEqual(float(H.max()), 1.0 + 1e-4)
        self.assertLess(float(H[64, 64]), 1e-4)                   # Nyquist

    def test_mix_of_equal_branches_is_exact(self):
        x = img(1, 96)
        self.assertTrue(torch.equal(band_split.mix(x, x.clone(), 2), x))

    def test_lowpass_and_highpass_partition_the_image(self):
        a, b = img(1, 96), img(1, 96, 1)
        lp = lambda v: band_split.lowpass(v, 2)
        self.assertLess(float((band_split.mix(a, b, 2) - (lp(a) + b - lp(b))).abs().max()), 1e-5)
        const = torch.full((1, 3, 96, 96), 0.4)
        self.assertLess(float((lp(const) - 0.4).abs().max()), 1e-4)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            band_split.mix(img(1, 96), img(1, 64), 2)


class _FakeTrainer:
    def __init__(self, logdir):
        self.logdir, self.current_epoch, self.saved = logdir, 0, []

    def save_checkpoint(self, path, weights_only=False):
        self.saved.append(os.path.basename(path))
        open(path, "w").close()


class TestPeriodicCheckpoint(unittest.TestCase):
    def test_rules(self):
        cb = PeriodicCheckpoint(every_n_epochs=5, epochs=[1, 7])
        self.assertEqual([e for e in range(1, 12) if cb._should_save(e)], [1, 5, 7, 10])
        with self.assertRaises(ValueError):
            PeriodicCheckpoint(every_n_epochs=0, save_on_train_end=False)

    def test_run_saves_prunes_and_names_the_final_epoch_once(self):
        with tempfile.TemporaryDirectory() as d:
            tr = _FakeTrainer(d)
            cb = PeriodicCheckpoint(every_n_epochs=2, max_keep=2)
            for e in range(5):
                tr.current_epoch = e
                cb.on_train_epoch_end(tr, None)
            # PL 1.4.2 leaves current_epoch un-advanced at on_train_end.
            cb.on_train_end(tr, None)
            cb.on_train_end(tr, None)
            self.assertEqual(tr.saved, ["checkpoint_epoch_0002.ckpt", "checkpoint_epoch_0004.ckpt",
                                        "checkpoint_epoch_0005.ckpt"])
            self.assertEqual(sorted(os.listdir(os.path.join(d, "checkpoints_periodic"))),
                             ["checkpoint_epoch_0004.ckpt", "checkpoint_epoch_0005.ckpt"])


class TestTeacherDataset(unittest.TestCase):
    def test_fingerprint_sees_content_and_order(self):
        from ldm_ftmse.data_teacher import paths_fingerprint as fp
        self.assertEqual(fp(["a", "b"]), fp(["a", "b"]))
        self.assertNotEqual(fp(["a", "b"]), fp(["b", "a"]))
        self.assertNotEqual(fp(["a", "b"]), fp(["a"]))

    def test_crop_overrides_are_refused(self):
        from ldm_ftmse.data_teacher import FileListSRTeacherTrain
        for bad in (dict(min_crop_f=0.5), dict(max_crop_f=0.9), dict(random_crop=True)):
            with self.assertRaises(ValueError, msg=str(bad)):
                FileListSRTeacherTrain("nope.txt", "nope", size=256, **bad)


if __name__ == "__main__":
    unittest.main()
