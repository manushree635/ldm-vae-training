from __future__ import annotations
"""distillation_nerc's data pipeline: a plain squashing resize, no cropping.

WHAT NERC ACTUALLY DOES (distillation_nerc/lightning_data.py:73-81):

    transforms.Compose([
        transforms.Resize((size, size)),                       # SQUASH to square
        transforms.ToTensor(),                                 # -> [0,1], CHW
        transforms.Normalize([0.5]*3, [0.5]*3),                # -> [-1,1]
    ])

torchvision's Resize with a (h, w) tuple ignores aspect ratio entirely -- a
735x1102 source is squashed straight to 256x256. There is no crop anywhere.

WHAT LDM DOES INSTEAD (ldm/data/imagenet.py:347-358):

    crop_side_len = min(H, W) * U(min_crop_f, max_crop_f)
    Center/RandomCrop(crop_side_len, crop_side_len)            # aspect-PRESERVING square crop
    SmallestMaxSize(size, cv2.INTER_AREA)                      # then resize

So upstream throws away everything outside a centred square and keeps the
subject's proportions; nerc keeps the whole frame and distorts it. On portrait
LAION faces (735x1102 here) that is a large difference: the crop discards ~33%
of the image height, while the squash compresses faces vertically by ~1.5x.

This class reproduces nerc's version inside LDM's data contract. It emits the
same dict keys ImageNetSR does ("image", "LR_image", "file_path_") in the same
HWC float32 [-1,1] layout main.py's get_input expects, so nothing downstream
changes.

SOURCE DATA. Pointed at filelists/subset_20k (the ORIGINAL uncropped images,
e.g. 735x1102), NOT subset_20k_rcrop (pre-cropped to 512x512). Using rcrop would
defeat the purpose -- those files have already had the crop baked in, so a
"plain resize" over them is still a crop, just someone else's. The 19k/1k split
is identical to every other arm, so only the preprocessing differs.

NOT TEACHER-COMPATIBLE. The precomputed teacher reconstructions are rendered
from subset_20k_rcrop and align only with a full-frame crop of those files. A
squash-resized uncropped source does not align with them, so this pipeline
cannot feed a steerable run without regenerating the teacher set. GAN-only arms
(no steerable term, no teacher) are the cheap use.
"""

import numpy as np
import PIL.Image
from torch.utils.data import Dataset

_INTERP = {
    "bilinear": PIL.Image.BILINEAR,   # torchvision Resize's default, i.e. nerc's
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
    "nearest": PIL.Image.NEAREST,
    "area": PIL.Image.BOX,
}


class _PlainResizeBase(Dataset):
    def __init__(self, images_list_file, size=256, downscale_f=4,
                 interpolation="bilinear", **_ignored):
        with open(images_list_file, "r") as f:
            self.paths = [p for p in f.read().splitlines() if p]
        self.size = int(size)
        self.LR_size = int(self.size / downscale_f)
        if interpolation not in _INTERP:
            raise ValueError("interpolation must be one of {}, got {!r}".format(
                sorted(_INTERP), interpolation))
        self.interp = _INTERP[interpolation]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        im = PIL.Image.open(path)
        if im.mode != "RGB":
            im = im.convert("RGB")

        # The whole point: (size, size) squashes, it does not preserve aspect.
        hr = im.resize((self.size, self.size), self.interp)
        lr = hr.resize((self.LR_size, self.LR_size), self.interp)

        return {
            "file_path_": path,
            # HWC float32 in [-1,1] -- matches ImageNetSR, which is what
            # AutoencoderKL.get_input permutes to NCHW.
            "image": (np.array(hr).astype(np.float32) / 127.5 - 1.0),
            "LR_image": (np.array(lr).astype(np.float32) / 127.5 - 1.0),
        }


class PlainResizeTrain(_PlainResizeBase):
    """nerc-style squash resize over an explicit file list."""


class PlainResizeValidation(_PlainResizeBase):
    """nerc-style squash resize over an explicit file list."""
