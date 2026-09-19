"""Turn a folder of arbitrary images into the 512x512 square set the teacher configs train on.

    python scripts/make_rcrop_dataset.py --src_dir $LDM_DATA/subset_20k \
        --out_dir $LDM_DATA/subset_20k_rcrop --workers 8

Each source image gets ONE fixed random 512x512 crop at native resolution (no
resize; images smaller than 512 on a side are zero-padded first), written as
<stem>_rcrop.jpg. The crop is seeded per image (seed + index in the sorted
listing), so re-running reproduces the same set.

WHY the teacher configs need this and cannot use the raw folder: the precomputed
teacher reconstruction is a full-frame 512x512 image, and the student must see
exactly the pixels it was computed from. On a square 512 source, upstream's
ImageNetSR pipeline at min_crop_f = max_crop_f = 1.0 reduces to "512 -> area
downsample to 256", so student and teacher are aligned by construction. On
non-square sources the two geometries disagree (ldm_ftmse/data_teacher.py).

Same procedure as distillation_nerc/data/data_prep/make_random_crop_subset.py.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import torch
from PIL import Image
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_image_paths(image_dir):
    paths = []
    for root, _dirs, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                p = os.path.join(root, fname)
                if os.path.getsize(p) > 0:
                    paths.append(p)
    return sorted(paths)


def _one(job):
    idx, src, dst, size, seed, quality = job
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return "skipped"
    try:
        im = Image.open(src).convert("RGB")
        w, h = im.size
        if w < size or h < size:
            pw, ph = max(0, size - w), max(0, size - h)
            im = TF.pad(im, padding=[pw // 2, ph // 2, pw - pw // 2, ph - ph // 2], fill=0)
        torch.manual_seed(seed + idx)
        i, j, ch, cw = RandomCrop.get_params(im, (size, size))
        TF.crop(im, i, j, ch, cw).save(dst, quality=quality, subsampling=0)
        return "ok"
    except Exception as e:                      # one unreadable file must not kill the run
        return "FAILED {}: {}".format(src, e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--suffix", default="_rcrop")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    paths = collect_image_paths(a.src_dir)
    if not paths:
        raise SystemExit("no images under {}".format(a.src_dir))
    jobs = [(i, p, os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + a.suffix + ".jpg"),
             a.crop_size, a.seed, a.quality) for i, p in enumerate(paths)]
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for n, r in enumerate(ex.map(_one, jobs, chunksize=64), 1):
            if r.startswith("FAILED"):
                counts["failed"] += 1
                print(r, flush=True)
            else:
                counts[r] += 1
            if n % 2000 == 0:
                print("  {}/{}".format(n, len(jobs)), flush=True)
    print("{} images -> {}   {}".format(len(jobs), a.out_dir, counts))
    if counts["failed"]:
        raise SystemExit("{} images failed; make_rcrop_filelists.py will report them as missing".format(counts["failed"]))


if __name__ == "__main__":
    main()
