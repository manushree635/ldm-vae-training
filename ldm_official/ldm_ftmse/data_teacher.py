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

import os

import numpy as np
from PIL import Image

from .data import _FileListSR


class _FileListSRTeacher(_FileListSR):
    def __init__(self, images_list_file, teacher_dir, *,
                 teacher_key="teacher", teacher_ext=".png", teacher_size=512,
                 verify_teacher=True, **kwargs):
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
        super().__init__(images_list_file, **kwargs)   # calls get_base() first

        if not os.path.isdir(teacher_dir):
            raise FileNotFoundError("teacher_dir does not exist: {}".format(teacher_dir))
        if verify_teacher:
            missing = [p for p in self.base.paths if not os.path.exists(self._teacher_path(p))]
            if missing:
                raise FileNotFoundError(
                    "{} of {} teacher recons missing under {} (e.g. {})".format(
                        len(missing), len(self.base.paths), teacher_dir,
                        [self._teacher_path(m) for m in missing[:3]]))

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
        return example


class FileListSRTeacherTrain(_FileListSRTeacher):
    """Teacher-paired train split."""


class FileListSRTeacherValidation(_FileListSRTeacher):
    """Teacher-paired validation split."""
