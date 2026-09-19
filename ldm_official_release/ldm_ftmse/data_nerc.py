"""distillation_nerc's data pipeline, reproduced for the LDM harness.

nerc (lightning_data.FaceDataModule + MultiViewDataset, datasets.FaceDataset):
  * paths  sorted(os.walk(subset_20k)) of .jpg/.jpeg/.png/.webp, capped at 20000
           (a no-op: the dir holds exactly 20000), then
           torch.utils.data.random_split([19000, 1000], generator=manual_seed(42)).
           The two lists in filelists/nerc_subset_20k/ were written by running
           nerc's own functions in its own torch (2.7), so the split is identical.
  * student  Image.open(src).convert("RGB") -> Resize((256,256)) -> ToTensor
             -> Normalize(0.5, 0.5): an anisotropic SQUASH of the full frame, no crop,
             no flip.
  * teacher  <teacher_dir>/<stem>.png .convert("RGB") -> Resize((512,512)) (identity
             on the 512 PNGs) -> ToTensor -> Normalize(0.5, 0.5).

The only change is layout: the LDM model wants HWC float32 in [-1, 1] under
"image" (AutoencoderKL.get_input permutes to NCHW), and the steerable model reads
the teacher from "teacher". torchvision's Resize on a PIL image is PIL's bilinear
resize in both torchvision 0.12 (here) and nerc's, so the pixels match.
"""
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _nerc_transform(size):
    return transforms.Compose([
        transforms.Resize((int(size), int(size))),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class _NercTeacherBase(Dataset):
    def __init__(self, images_list_file, teacher_dir, size=256, teacher_size=512,
                 teacher_key="teacher"):
        with open(images_list_file) as f:
            self.paths = [l.strip() for l in f if l.strip()]
        if not os.path.isdir(teacher_dir):
            raise FileNotFoundError("teacher_dir does not exist: {}".format(teacher_dir))
        self.teacher_dir = teacher_dir
        self.teacher_key = teacher_key
        self.t_student = _nerc_transform(size)
        self.t_teacher = _nerc_transform(teacher_size)
        missing = [p for p in self.paths[:200] if not os.path.isfile(self._teacher_path(p))]
        if missing:
            raise FileNotFoundError("{} of the first 200 images have no teacher recon in {} "
                                    "(e.g. {})".format(len(missing), teacher_dir, missing[0]))

    def _teacher_path(self, src):
        return os.path.join(self.teacher_dir, os.path.splitext(os.path.basename(src))[0] + ".png")

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _hwc(t):
        return t.permute(1, 2, 0).numpy().astype(np.float32)

    def __getitem__(self, i):
        src = self.paths[i]
        with Image.open(src) as im:
            student = self.t_student(im.convert("RGB"))
        with Image.open(self._teacher_path(src)) as im:
            teacher = self.t_teacher(im.convert("RGB"))
        return {"image": self._hwc(student), self.teacher_key: self._hwc(teacher),
                "file_path_": src}


class NercTeacherTrain(_NercTeacherBase):
    pass


class NercTeacherValidation(_NercTeacherBase):
    pass
