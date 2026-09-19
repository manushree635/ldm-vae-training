"""Precompute the 512px teacher reconstruction of every image, once, as PNGs.

    python scripts/precompute_teacher_recons.py \
        --src_dir $LDM_DATA/subset_20k_rcrop \
        --out_dir $LDM_DATA/subset_20k_rcrop_teacher_recons_512_png

The steerable term distils the student (decoding at 256) towards what the TEACHER
VAE reconstructs at 512. Running the teacher inside the training loop costs an
encode+decode at 512 per step (~7x slower end to end), and with latent_mode=mean
its output is deterministic -- so it is computed once here instead.

Per image:  RGB -> Resize((512, 512)) -> [-1, 1] -> encode -> posterior MEAN
            -> decode -> clamp -> 8-bit PNG named <source stem>.png
which is the naming ldm_ftmse/data_teacher.py and data_nerc.py look up.

Note Resize((512,512)) is an anisotropic squash on a non-square source. That is
what the `nercmatch` configs expect (their student input is squashed the same
way). The FileListSRTeacher configs instead need square 512 sources, where the
resize is the identity -- build those with make_rcrop_dataset.py first.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class _Images(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--teacher", default="stabilityai/sd-vae-ft-mse",
                    help="any diffusers AutoencoderKL repo or local directory")
    ap.add_argument("--subfolder", default="")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--no_amp", action="store_true", help="fp32 instead of fp16 autocast")
    a = ap.parse_args()

    from diffusers import AutoencoderKL

    paths = sorted(os.path.join(r, f) for r, _d, fs in os.walk(a.src_dir) for f in fs
                   if f.lower().endswith(IMAGE_EXTS) and os.path.getsize(os.path.join(r, f)) > 0)
    if not paths:
        raise SystemExit("no images under {}".format(a.src_dir))
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(set(stems)) != len(stems):
        raise SystemExit("duplicate file stems under {} -- recons are keyed by stem".format(a.src_dir))
    os.makedirs(a.out_dir, exist_ok=True)
    have = set(os.listdir(a.out_dir))
    todo = [p for p, s in zip(paths, stems) if s + ".png" not in have]
    print("{} images, {} already done, {} to do".format(len(paths), len(paths) - len(todo), len(todo)), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"subfolder": a.subfolder} if a.subfolder else {}
    vae = AutoencoderKL.from_pretrained(a.teacher, **kw).eval().to(dev)
    use_amp = dev == "cuda" and not a.no_amp

    loader = DataLoader(_Images(todo, a.size), batch_size=a.batch_size, num_workers=a.num_workers)
    done = 0
    with torch.no_grad():
        for x, idx in loader:
            with torch.autocast(device_type=dev, dtype=torch.float16, enabled=use_amp):
                rec = vae.decode(vae.encode(x.to(dev)).latent_dist.mean).sample
            rec = ((rec.float().clamp(-1, 1) * 0.5 + 0.5) * 255.0).round().to(torch.uint8)
            rec = rec.permute(0, 2, 3, 1).cpu().numpy()
            for r, i in zip(rec, idx.tolist()):
                stem = os.path.splitext(os.path.basename(todo[i]))[0]
                Image.fromarray(np.ascontiguousarray(r), mode="RGB").save(
                    os.path.join(a.out_dir, stem + ".png"), compress_level=1)
            done += len(idx)
            if done % (a.batch_size * 100) == 0:
                print("  {}/{}".format(done, len(todo)), flush=True)

    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump({"teacher": a.teacher, "subfolder": a.subfolder, "src_dir": os.path.abspath(a.src_dir),
                   "size": a.size, "latent_mode": "mean", "amp": use_amp, "n": len(paths)}, f, indent=2)
    print("done -> {}".format(a.out_dir))


if __name__ == "__main__":
    main()
