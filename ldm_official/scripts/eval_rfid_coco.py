"""Score LDM-trained checkpoints with distillation_nerc's own rFID harness.

Reuses vae_bench.bench.stage_recon unmodified rather than reimplementing FID, so
the numbers land on the same scale as the existing recon.json table (same COCO
images in the same order, same Resize->CenterCrop, same fp16 load / fp32 decode,
same posterior-mean latent, same torchmetrics FID instance-per-arm).

Extra arms are registered into bench.VAES at runtime; bench.py itself is not
edited. `sd-vae` is included as a control -- it must reproduce its published
10.483, otherwise the harness is not being driven identically and the new rows
mean nothing.

    python scripts/eval_rfid_coco.py --arm ldm-ftema-ep0=models/..._ep0_diffusers
"""

import argparse
import os
import sys
from pathlib import Path

NERC = "/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/distillation_nerc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="NAME=/path/to/diffusers_dir (repeatable)")
    ap.add_argument("--control", default="sd-vae",
                    help="comma-separated existing bench arms to include; "
                         "'sd-vae' doubles as the harness check ('' to skip)")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    sys.path.insert(0, NERC)
    import torch
    from vae_bench import bench

    arms = []
    for name in filter(None, (n.strip() for n in args.control.split(","))):
        if name not in bench.VAES:
            raise SystemExit(f"unknown bench arm {name!r}; known: {sorted(bench.VAES)}")
        arms.append(name)
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm needs NAME=PATH, got {spec!r}")
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise SystemExit(f"not a directory: {path}")
        bench.VAES[name] = (path, None)
        arms.append(name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  arms={arms}  n={args.n}  size={args.size}", flush=True)
    if device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    bench.stage_recon(device, out, args.n, args.size, args.batch_size, arms, args.tag)
    print("Done", flush=True)


if __name__ == "__main__":
    main()
