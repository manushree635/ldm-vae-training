"""Reconstruct one image through every arm, using vae_bench's own conventions.

fp16 load / fp32 decode / posterior MEAN latent / Resize->CenterCrop, all lifted
from vae_bench.bench so these pictures correspond to the rFID table rather than
to a separate ad-hoc pipeline.
"""
import argparse, os, sys
from pathlib import Path

NERC = "/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/distillation_nerc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--arm", action="append", default=[], help="NAME=path_or_repo[,subfolder]")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--squash", action="store_true",
                    help="Resize((S,S)) like distillation_nerc's inference.py, "
                         "instead of vae_bench's Resize(S)->CenterCrop(S)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, NERC)
    import torch
    from PIL import Image
    from vae_bench import bench

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.squash:
        # distillation_nerc/inference.py: Resize((S,S)) -- squashes aspect ratio.
        from torchvision import transforms
        tf = transforms.Compose([transforms.Resize((args.size, args.size)),
                                 transforms.ToTensor()])
        x01 = tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    else:
        x01 = bench.load_batch([Path(args.image)], args.size).to(device)


    def save(t, name):
        bench.save_png(t[0].cpu(), out / f"{name}.png")

    save(x01, "00_original")
    print(f"original -> {out}/00_original.png  ({args.size}px)")

    for i, spec in enumerate(args.arm, start=1):
        name, _, loc = spec.partition("=")
        repo, sub = (loc.split(",", 1) + [None])[:2] if "," in loc else (loc, None)
        bench.VAES[name] = (repo, sub)
        vae = bench.load_vae(name, device)
        with torch.no_grad():
            z = vae.encode((x01 * 2 - 1).half()).latent_dist.mean * bench.SCALE
        rec = bench.decode_fp32(vae, z / bench.SCALE, name)
        mse = ((rec - x01) ** 2).flatten(1).mean(1).clamp_min(1e-12)
        psnr = float((10 * torch.log10(1.0 / mse)).item())
        save(rec, f"{i:02d}_{name}")
        print(f"{name:<22} PSNR={psnr:6.2f}  -> {i:02d}_{name}.png")
        del vae; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
