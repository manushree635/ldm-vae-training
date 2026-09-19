from __future__ import annotations
"""Teacher-paired dataset: upstream's ImageNetSR pipeline plus a precomputed
512px teacher reconstruction for each image.

The whole design turns on ALIGNMENT. Upstream's __getitem__ picks

    crop_side_len = min_side * uniform(min_crop_f, max_crop_f)

and crops at a random position, freshly on every call. A precomputed full-frame
teacher reconstruction would then correspond to a different region of the image
than the student's input, and nothing would report it -- the steerable term would
simply never go down. So this class FORCES min_crop_f = max_crop_f = 1.0 and
random_crop = False, and refuses to be overridden.

On a 512x512 source that makes the crop an identity and the pipeline exactly
"512 -> area-downsample to 256", so the student sees precisely the pixels the
teacher was computed from.

This also only works because the source is SQUARE. The teacher recons were built
with Resize((512,512)) -- an anisotropic squash of the full frame -- while
ImageNetSR does CenterCrop(min_side) then resize, an aspect-preserving crop. On
subset_20k (166/200 images non-square by >5%) those two geometries disagree
completely. On subset_20k_rcrop they coincide.
"""

import hashlib
import json
import os

import numpy as np
from PIL import Image

from .autoencoder import LATENTS_KEY
from .data import _FileListSR


def paths_fingerprint(paths):
    """Stable id for a filelist's contents AND order.

    The latent cache is addressed by row index, so a filelist that gained,
    lost or reordered a line would silently pair image i with image j's
    moments -- a corruption that trains perfectly happily and is invisible in
    the loss. This is what makes that a startup error instead.
    """
    h = hashlib.sha256()
    for q in paths:
        h.update(q.encode("utf-8"))
        h.update(b"\\n")
    return h.hexdigest()


class _FileListSRTeacher(_FileListSR):
    def __init__(self, images_list_file, teacher_dir, *,
                 teacher_key="teacher", teacher_ext=".png", teacher_size=512,
                 verify_teacher=True, latents_file=None, **kwargs):
        for k, want in (("min_crop_f", 1.0), ("max_crop_f", 1.0), ("random_crop", False)):
            if k in kwargs and kwargs[k] != want:
                raise ValueError(
                    "{}={!r} would break teacher alignment; it is forced to {!r}. "
                    "A precomputed full-frame teacher only corresponds to a "
                    "full-frame, deterministic student crop.".format(k, kwargs[k], want))
            kwargs[k] = want

        self.teacher_dir = teacher_dir
        self.teacher_key = teacher_key
        self.teacher_ext = teacher_ext
        self.teacher_size = int(teacher_size)
        self.latents_file = latents_file
        self._latents_mm = None            # opened lazily, per worker -- see __getitem__
        super().__init__(images_list_file, **kwargs)   # calls get_base() first

        if not os.path.isdir(teacher_dir):
            raise FileNotFoundError("teacher_dir does not exist: {}".format(teacher_dir))
        if verify_teacher:
            # One listdir beats len(paths) stat()s: at 20k the difference is
            # invisible, but a 1M-image filelist would stat 1M times on GPFS,
            # per rank, on every job start.
            have = set(os.listdir(teacher_dir))
            missing = [p for p in self.base.paths
                       if os.path.basename(self._teacher_path(p)) not in have]
            if missing:
                raise FileNotFoundError(
                    "{} of {} teacher recons missing under {} (e.g. {})".format(
                        len(missing), len(self.base.paths), teacher_dir,
                        [self._teacher_path(m) for m in missing[:3]]))

        if self.latents_file is not None:
            self._verify_latents()

    def _verify_latents(self):
        """Refuse a cache that does not provably belong to this filelist.

        Caching the encoder is only sound because this class forces a
        deterministic crop (see the checks in __init__). Assert that here too
        rather than trusting the caller: a subclass that relaxed it would make
        every cached latent wrong, and nothing downstream would notice.
        """
        # NB ImageNetSR stores `random_crop` inverted, as `center_crop` -- there
        # is no self.random_crop to check.
        for k, want in (("min_crop_f", 1.0), ("max_crop_f", 1.0), ("center_crop", True)):
            got = getattr(self, k, None)
            if got != want:
                raise ValueError(
                    "latents_file requires a deterministic pipeline; {}={!r}, "
                    "expected {!r}. Cached moments would not match the crop the "
                    "student is trained on.".format(k, got, want))

        meta_path = self.latents_file + ".json"
        if not os.path.exists(self.latents_file) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                "latents cache incomplete: expected {} and {}. Build it with "
                "scripts/precompute_latents.py.".format(self.latents_file, meta_path))
        with open(meta_path) as f:
            meta = json.load(f)

        n = len(self.base.paths)
        if meta.get("n") != n:
            raise ValueError("latents cache holds {} rows, filelist has {}".format(
                meta.get("n"), n))
        target = "{}.{}".format(type(self).__module__, type(self).__name__)
        built_by = meta.get("dataset_target")
        if built_by is not None and built_by.split(".")[-2] != target.split(".")[-2]:
            raise ValueError(
                "latents cache was built by {} but this dataset is {}. Those "
                "modules use DIFFERENT input geometry (data_teacher centre-crops "
                "to the short side; data_squash resizes the whole frame), so the "
                "cached moments do not correspond to the images this run feeds "
                "the decoder. Rebuild with scripts/precompute_latents.py."
                .format(built_by, target))

        want = paths_fingerprint(self.base.paths)
        if meta.get("paths_sha256") != want:
            raise ValueError(
                "latents cache was built from a different filelist (or a "
                "different order): {} has sha256 {}, this dataset {}. Rebuild "
                "with scripts/precompute_latents.py.".format(
                    meta.get("images_list_file"), meta.get("paths_sha256"), want))

    def _moments(self):
        # Opened on first use rather than in __init__ so each DataLoader worker
        # gets its own handle after fork, and so the mmap is never pickled.
        if self._latents_mm is None:
            self._latents_mm = np.load(self.latents_file, mmap_mode="r")
        return self._latents_mm

    def _teacher_path(self, src_path):
        stem = os.path.splitext(os.path.basename(src_path))[0]
        return os.path.join(self.teacher_dir, stem + self.teacher_ext)

    def __getitem__(self, i):
        example = super().__getitem__(i)          # "image" HWC [-1,1], "LR_image", "file_path_"
        path = self._teacher_path(example["file_path_"])
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.uint8)
        if arr.shape[:2] != (self.teacher_size, self.teacher_size):
            raise ValueError("teacher {} is {}, expected {}x{}".format(
                path, arr.shape[:2], self.teacher_size, self.teacher_size))
        # HWC float32 in [-1,1] -- get_input() permutes NHWC->NCHW, so match "image".
        example[self.teacher_key] = (arr / 127.5 - 1.0).astype(np.float32)
        if self.latents_file is not None:
            # .copy(): the mmap row is read-only, and torch.from_numpy on a
            # read-only array warns. 32 KB, so the copy costs nothing.
            example[LATENTS_KEY] = np.asarray(self._moments()[i]).copy()
        return example


class FileListSRTeacherTrain(_FileListSRTeacher):
    """Teacher-paired train split."""


class FileListSRTeacherValidation(_FileListSRTeacher):
    """Teacher-paired validation split."""
