"""Train/val file lists for a teacher-paired set, keeping only images that HAVE a teacher.

    python scripts/make_filelists_teacher.py \
        --data_dir    $LDM_DATA/laion500k_oi500k_merged \
        --teacher_dir $LDM_DATA/laion500k_oi500k_teacher_recons_512_png \
        --out_dir     filelists/laion500k_oi500k_1M --val_n 2000

Two departures from make_filelists.py, both forced by the 1M set:

1. INTERSECTION. The merged dir holds 1,000,001 images, the teacher dir
   1,000,000 -- at least one image has no recon. data_teacher.py's
   verify_teacher would abort the run on it, so drop it here instead.

2. ABSOLUTE val size, not a fraction. make_filelists.py's 0.95 split would put
   50,000 images in validation; validation runs the autoencoder TWICE per batch
   (once under ema_scope) plus three loss evaluations, so that would add ~15% to
   every epoch for a number nobody reads at that precision. --val_n 2000 keeps
   it ~0.2%.

Split semantics otherwise match make_filelists.py exactly: sorted absolute
paths, torch.randperm with seed 42, so it is reproducible and consistent with
the 20k lists.
"""
import argparse, os
import torch

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--teacher_dir", required=True)
    ap.add_argument("--teacher_ext", default=".png")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"listing {a.data_dir} ...", flush=True)
    paths = sorted(
        os.path.join(r, f)
        for r, _d, fs in os.walk(a.data_dir) for f in fs
        if f.lower().endswith(IMAGE_EXTS)
    )
    print(f"  {len(paths):,} images", flush=True)

    print(f"listing {a.teacher_dir} ...", flush=True)
    have = set(os.listdir(a.teacher_dir))
    print(f"  {len(have):,} teacher recons", flush=True)

    kept, dropped = [], []
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        (kept if stem + a.teacher_ext in have else dropped).append(p)
    print(f"  kept {len(kept):,}; dropped {len(dropped):,} with no teacher"
          + (f" (e.g. {[os.path.basename(d) for d in dropped[:3]]})" if dropped else ""))

    # Zero-byte check only on what survived -- os.path.getsize on 1M files is
    # expensive and the teacher intersection already removed the likely culprits.
    kept = [p for p in kept if os.path.getsize(p) > 0]
    print(f"  {len(kept):,} after dropping zero-byte files", flush=True)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(len(kept), generator=g).tolist()
    val = [kept[i] for i in perm[:a.val_n]]
    train = [kept[i] for i in perm[a.val_n:]]

    os.makedirs(a.out_dir, exist_ok=True)
    for name, subset in (("train.txt", train), ("val.txt", val)):
        out = os.path.join(a.out_dir, name)
        with open(out, "w") as f:
            f.write("\n".join(subset) + "\n")
        print(f"{out}: {len(subset):,} images")


if __name__ == "__main__":
    main()
