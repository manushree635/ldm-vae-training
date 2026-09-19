"""Score a trained checkpoint on a file list: PSNR, LPIPS (VGG) and reconstruction FID.

    source scripts/env.sh
    python scripts/eval_recon.py --ckpt logs/<run>/checkpoints/last.ckpt --use_ema \
        --filelist filelists/subset_20k_rcrop/val.txt --out logs/<run>/eval_val.json

    # the starting point, for the row to compare against
    python scripts/eval_recon.py --ckpt models/ftmse_vae_ldm.ckpt \
        --filelist filelists/subset_20k_rcrop/val.txt

Self-contained: it loads the LDM checkpoint directly (no export step) and uses the
same preprocessing as validation -- centre-crop to the short side, INTER_AREA
resize to --size, x/127.5-1 -- and the posterior MEAN, so it is deterministic.

Read the numbers with care:
  * rFID is biased high at small n. 1,000 val images are fine for ranking runs
    against each other on the SAME list; they are not comparable to published
    50k-image figures.
  * val/rec_loss, which main.py uses to keep its top-3 checkpoints, is
    ANTI-correlated with rFID on GAN runs. Pick checkpoints with this script.

Without --config the checkpoint is loaded into the stock kl-f8 AutoencoderKL,
which covers every model class here that keeps that layout. The *dual* configs
do not: their second decoder tail would be dropped and you would score branch 1
alone, so for those pass the training YAML and the model is built from its
`model.target` instead (decode() then returns the band-mixed output):

    python scripts/eval_recon.py --config configs/12_dual_branch_1M.yaml \
        --ckpt logs/<run>/checkpoints/last.ckpt --use_ema --filelist ...
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "latent-diffusion"), os.path.join(ROOT, "taming-transformers")]

DDCONFIG = dict(double_z=True, z_channels=4, resolution=256, in_channels=3, out_ch=3, ch=128,
                ch_mult=[1, 2, 4, 4], num_res_blocks=2, attn_resolutions=[], dropout=0.0)


def load_model(ckpt, use_ema, config=None):
    from ldm.models.autoencoder import AutoencoderKL
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    if use_ema:
        # LitEma keeps each shadow as a buffer named model_ema.<param name minus dots>
        shadows = {k[len("model_ema."):]: v for k, v in sd.items()
                   if k.startswith("model_ema.") and k.split(".")[-1] not in ("decay", "num_updates")}
        if not shadows:
            raise SystemExit("--use_ema given but {} has no model_ema.* buffers".format(ckpt))
        n = 0
        for k in list(sd):
            s = k.replace(".", "")
            if s in shadows:
                sd[k] = shadows[s]
                n += 1
        print("swapped in {} EMA tensors".format(n))
    sd = {k: v for k, v in sd.items() if not k.startswith(("loss.", "model_ema."))}
    if config is not None:
        # Build the run's own model class (needed when the decoder is not the stock one).
        # ckpt_path is dropped so the init checkpoint is never read; the loss is kept
        # because some classes consult it in decode() (dual-branch: steerable_order).
        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        cfg = OmegaConf.load(config).model
        cfg.params.pop("ckpt_path", None)
        model = instantiate_from_config(cfg)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith(("loss.", "model_ema."))]
        if "decoder_b2" in dict(model.named_children()) and not any(k.startswith("decoder_b2.") for k in sd):
            raise SystemExit("{} builds a second decoder branch but {} has no decoder_b2.* tensors"
                             .format(config, ckpt))
    else:
        if any(k.startswith("decoder_b2.") for k in sd):
            raise SystemExit("{} is a dual-branch checkpoint; pass --config <its training yaml>, "
                             "otherwise only branch 1 would be scored".format(ckpt))
        model = AutoencoderKL(ddconfig=DDCONFIG, lossconfig={"target": "torch.nn.Identity"}, embed_dim=4)
        missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit("checkpoint is missing {} model tensors, e.g. {}".format(len(missing), missing[:3]))
    if unexpected:
        print("note: {} tensors in the checkpoint are not part of the stock VAE and were ignored "
              "(e.g. {})".format(len(unexpected), unexpected[:2]))
    return model.eval()


def load_image(path, size):
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w = im.shape[:2]
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    im = im[top:top + s, left:left + s]
    if s != size:
        im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--filelist", required=True)
    ap.add_argument("--config", default=None,
                    help="training YAML; build the model from its model.target (REQUIRED for dual-branch runs)")
    ap.add_argument("--use_ema", action="store_true", help="score the EMA weights (the *EMA model classes)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=0, help="first n images only (0 = all)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_fid", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    with open(a.filelist) as f:
        paths = [l.strip() for l in f if l.strip()]
    if a.n:
        paths = paths[:a.n]
    # absolute BEFORE the chdir below
    ckpt, filelist = os.path.abspath(a.ckpt), os.path.abspath(a.filelist)
    out = os.path.abspath(a.out) if a.out else None
    config = os.path.abspath(a.config) if a.config else None
    # taming resolves its LPIPS weights relative to the cwd; latent-diffusion/ has them.
    os.chdir(os.path.join(ROOT, "latent-diffusion"))
    from taming.modules.losses.lpips import LPIPS

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, a.use_ema, config).to(dev)
    lpips = LPIPS().eval().to(dev)
    fid = None
    if not a.no_fid:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048).to(dev)

    psnrs, lps = [], []
    with torch.no_grad():
        for i in range(0, len(paths), a.batch_size):
            x = torch.stack([load_image(p, a.size) for p in paths[i:i + a.batch_size]]).to(dev)
            rec = model.decode(model.encode(x).mode()).clamp(-1, 1)
            mse = ((x - rec) ** 2).flatten(1).mean(1) / 4.0          # on a [0, 1] scale
            psnrs += (-10 * torch.log10(mse.clamp_min(1e-12))).tolist()
            lps += lpips(x, rec).flatten().tolist()
            if fid is not None:
                fid.update(((x + 1) * 127.5).round().to(torch.uint8), real=True)
                fid.update(((rec + 1) * 127.5).round().to(torch.uint8), real=False)
            if (i // a.batch_size) % 20 == 0:
                print("  {}/{}".format(min(i + a.batch_size, len(paths)), len(paths)), flush=True)

    res = {"ckpt": ckpt, "use_ema": a.use_ema, "filelist": filelist,
           "n": len(paths), "size": a.size,
           "psnr": float(np.mean(psnrs)), "lpips_vgg": float(np.mean(lps)),
           "rfid": float(fid.compute()) if fid is not None else None}
    print(json.dumps(res, indent=2))
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
