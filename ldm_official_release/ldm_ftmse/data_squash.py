from __future__ import annotations
"""Teacher-paired dataset whose student input matches the teacher's GEOMETRY.

WHY THIS EXISTS. data_teacher.py inherits upstream's ImageNetSR pipeline:

    CenterCrop(min_side)  ->  SmallestMaxSize(size, INTER_AREA)

which is ASPECT-PRESERVING -- it throws away everything outside a centred
square. That is correct only when the source is already square, which is why
the 20k configs point at subset_20k_rcrop rather than subset_20k.

distillation_nerc builds both halves with an ANISOTROPIC SQUASH instead:

    precompute_teacher_recons.py:72   transforms.Resize((512, 512))
    lightning_data.py:302             transforms.Resize((256, 256))

so its teacher recons correspond to the WHOLE frame, squashed. Pairing those
teachers with ImageNetSR's centre-crop misaligns every non-square image. On
split_0 that is 243 of 300 sampled images (>5% off square, up to 3:1), and on a
580x1740 example the two geometries share only 33% of the frame and differ by a
mean of 43.9/255 -- they are different pictures. The steerable term would simply
never go down, with nothing reporting why.

So this class reproduces distillation_nerc's geometry exactly: one anisotropic
Resize((size, size)) of the full frame, via the same torchvision transform, and
NO crop. Value convention stays ldm_official's own -- HWC float32 in [-1, 1] via
`arr / 127.5 - 1.0`, which is algebraically what ToTensor + Normalize(0.5, 0.5)
produces.

It also drops upstream's `LR_image` branch, a 64px resize computed for every
sample that AutoencoderKL never consumes.

Determinism: there is no randomness at all here, so the latent cache is valid by
construction rather than by the forced-crop invariant data_teacher.py relies on.
"""

import os

import numpy as np
from PIL import Image
from torchvision import transforms

from .autoencoder import LATENTS_KEY
from .data_teacher import _FileListSRTeacher


class _FileListSquashTeacher(_FileListSRTeacher):
    def __init__(self, *args, teacher_from_source=False, **kwargs):
        # teacher_from_source=True: the steerable TARGET is the source image itself,
        # squashed to teacher_size x teacher_size with the same transform -- i.e. GT 512,
        # exactly the tensor precompute_teacher_recons.py fed the teacher VAE, instead of
        # that VAE's reconstruction. Built from the image already decoded for the student,
        # so no teacher files are read or verified (teacher_dir must still exist; point it
        # at the source images). Default False = the precomputed-teacher path, unchanged.
        self.teacher_from_source = bool(teacher_from_source)
        if self.teacher_from_source:
            kwargs["verify_teacher"] = False
        super().__init__(*args, **kwargs)
        # Same call distillation_nerc makes. torchvision's default interpolation
        # is BILINEAR, and PIL antialiases when downscaling -- do NOT swap this
        # for cv2 or a different filter: the teacher PNGs were produced through
        # this exact transform, and a different kernel reintroduces the
        # misalignment in miniature.
        self._squash = transforms.Resize((self.size, self.size))
        self._squash_teacher = transforms.Resize((self.teacher_size, self.teacher_size))

    def __getitem__(self, i):
        path = self.base.paths[i]
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(self._squash(img)).astype(np.uint8)
        example = {
            "file_path_": path,
            # float64 divide then cast, matching data_teacher.py's teacher branch
            # so student and teacher round identically.
            "image": (arr / 127.5 - 1.0).astype(np.float32),
        }

        if self.teacher_from_source:
            tpath = path
            tarr = np.array(self._squash_teacher(img)).astype(np.uint8)
        else:
            tpath = self._teacher_path(path)
            timg = Image.open(tpath)
            if timg.mode != "RGB":
                timg = timg.convert("RGB")
            tarr = np.array(timg).astype(np.uint8)
        if tarr.shape[:2] != (self.teacher_size, self.teacher_size):
            raise ValueError("teacher {} is {}, expected {}x{}".format(
                tpath, tarr.shape[:2], self.teacher_size, self.teacher_size))
        example[self.teacher_key] = (tarr / 127.5 - 1.0).astype(np.float32)

        if self.latents_file is not None:
            example[LATENTS_KEY] = np.asarray(self._moments()[i]).copy()
        return example


class FileListSquashTeacherTrain(_FileListSquashTeacher):
    """Squash-geometry train split."""


class FileListSquashTeacherValidation(_FileListSquashTeacher):
    """Squash-geometry validation split."""
