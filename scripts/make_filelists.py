"""Write the train/val file lists that taming's CustomTrain/CustomTest consume.

The split is deliberately byte-identical to the one distillation_nerc's
FaceDataModule produces for the same directory, so the LDM baseline and the
distillation runs are scored on exactly the same held-out images:

  * enumerate with os.walk, keep known image extensions, drop zero-byte files,
    sort the absolute paths                      (datasets.collect_image_paths)
  * perm = torch.randperm(N, generator=Generator().manual_seed(42))
    train = perm[:int(0.95 * N)], val = perm[int(0.95 * N):]
                                     (torch.utils.data.random_split semantics)

    python scripts/make_filelists.py \
        --data_dir $LDM_DATA/subset_20k \
        --out_dir filelists/subset_20k
"""

import argparse
import os

import torch

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_image_paths(image_dir):
    paths = []
    for root, _dirs, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(IMAGE_EXTS):
                p = os.path.join(root, fname)
                if os.path.getsize(p) <= 0:
                    continue
                paths.append(p)
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_split", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = collect_image_paths(args.data_dir)
    if not paths:
        raise SystemExit(f"no images under {args.data_dir}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(paths), generator=g).tolist()
    n_train = int(args.train_split * len(paths))
    train = [paths[i] for i in perm[:n_train]]
    val = [paths[i] for i in perm[n_train:]]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, subset in (("train.txt", train), ("val.txt", val)):
        out = os.path.join(args.out_dir, name)
        with open(out, "w") as f:
            f.write("\n".join(subset) + "\n")
        print(f"{out}: {len(subset):,} images")
    print(f"total {len(paths):,} from {args.data_dir} "
          f"(train_split={args.train_split}, seed={args.seed})")


if __name__ == "__main__":
    main()
