"""Map the subset_20k train/val split onto subset_20k_rcrop, preserving membership.

Do NOT regenerate the split with make_filelists.py for the rcrop set. That seeds
randperm off the SORTED path list, and the "_rcrop" suffix changes the sort order,
so ~5% of images would swap between train and val relative to the baseline run --
images the baseline trained on would land in the steerable run's validation set.
That is direct leakage, and it would quietly invalidate every comparison against
the existing emagan numbers.

Instead map one-for-one by filename stem, keeping line order:
    .../subset_20k/{stem}.jpg  ->  .../subset_20k_rcrop/{stem}_rcrop.jpg
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_lists", default=None,
                    help="directory holding the baseline train.txt / val.txt")
    ap.add_argument("--rcrop_dir", required=True,
                    help="directory of <stem>_rcrop.jpg crops (scripts/make_rcrop_dataset.py)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--suffix", default="_rcrop")
    args = ap.parse_args()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_lists = args.src_lists or os.path.join(here, "filelists", "subset_20k")
    os.makedirs(args.out_dir, exist_ok=True)

    total = missing = 0
    for name in ("train.txt", "val.txt"):
        src = os.path.join(src_lists, name)
        with open(src) as fh:
            paths = [p for p in fh.read().splitlines() if p]
        mapped, absent = [], []
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            q = os.path.join(args.rcrop_dir, stem + args.suffix + ".jpg")
            mapped.append(q)
            if not os.path.exists(q):
                absent.append(q)
        out = os.path.join(args.out_dir, name)
        with open(out, "w") as fh:
            fh.write("\n".join(mapped) + "\n")
        total += len(mapped)
        missing += len(absent)
        print("{}: {:,} paths -> {}{}".format(
            name, len(mapped), out,
            "  MISSING {} e.g. {}".format(len(absent), absent[:2]) if absent else ""))

    print("total {:,}, missing {}".format(total, missing))
    if missing:
        print("refusing to proceed with missing files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
