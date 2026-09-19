"""Precompute the frozen encoder's output once, so training stops recomputing it.

    source scripts/env.sh && cd latent-diffusion && python ../scripts/precompute_latents.py \
        --config ../configs/<cfg>.yaml --split train --out /path/moments_train.npy

Why this is sound rather than an approximation: the encoder and quant_conv are
frozen (ldm_ftmse/autoencoder.py), and ldm_ftmse/data_teacher.py FORCES
min_crop_f = max_crop_f = 1.0 and random_crop = False -- so `quant_conv(encoder(x_i))`
is the same tensor on every epoch, and upstream recomputes it twice per batch.

Writes `<out>` (fp32 .npy, N x 8 x 32 x 32) and `<out>.json`, which pins the
filelist the rows are ordered by. The cache is addressed by ROW INDEX, so a
filelist that later gains, loses or reorders a line would pair image i with
image j's moments -- training would look perfectly healthy and be wrong. The
sidecar turns that into a startup error.

ONE CAVEAT worth knowing. `benchmark: True` means cuDNN autotunes its conv
algorithms, so the encoder's output can differ in the last ULP between GPU
models. To claim bit-exactness against a specific run, build the cache on the
SAME node as that run -- scripts/run_ab_latentcache_scc.sh does exactly this.
"""
import argparse, json, os, sys, time

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "latent-diffusion"), os.path.join(_ROOT, "taming-transformers")]
from ldm.util import instantiate_from_config
from ldm_ftmse.data_teacher import paths_fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(args.config)

    data_cfg = OmegaConf.to_container(cfg.data.params[args.split], resolve=True)
    # Build the cache from a dataset that is NOT reading a cache.
    data_cfg["params"].pop("latents_file", None)
    ds = instantiate_from_config(data_cfg)
    paths = ds.base.paths
    n = len(paths)
    print(f"{args.split}: {n} images from {data_cfg['params']['images_list_file']}")

    model = instantiate_from_config(cfg.model).cuda().eval()

    # Shape is a property of ddconfig, not something to hardcode: 2*z_channels
    # (mean and logvar) at resolution / 2**(len(ch_mult)-1).
    dd = cfg.model.params.ddconfig
    zc, res = int(dd.z_channels) * 2, int(data_cfg["params"]["size"])
    lat = res // 2 ** (len(dd.ch_mult) - 1)
    print(f"moments shape per image: ({zc}, {lat}, {lat})  ->  "
          f"{n * zc * lat * lat * 4 / 2**30:.2f} GiB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    mm = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32,
                                   shape=(n, zc, lat, lat))

    # shuffle=False is load-bearing: rows are addressed by dataset index.
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    shuffle=False, drop_last=False)

    t0, done = time.time(), 0
    with torch.no_grad():
        for batch in dl:
            x = batch["image"]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
            x = x.cuda(non_blocking=True)
            moments = model.quant_conv(model.encoder(x))
            mm[done:done + moments.shape[0]] = moments.cpu().numpy()
            done += moments.shape[0]
            if done % (args.batch_size * 20) == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{n}  {rate:.1f} img/s  "
                      f"eta {(n - done) / rate / 60:.1f} min", flush=True)

    assert done == n, f"wrote {done} rows, expected {n}"
    mm.flush()
    del mm

    with open(args.out + ".json", "w") as f:
        json.dump({
            "images_list_file": data_cfg["params"]["images_list_file"],
            # The GEOMETRY fingerprint. A cache built through ImageNetSR's
            # centre-crop is NOT interchangeable with one built through
            # data_squash's anisotropic resize, and the config path alone cannot
            # tell them apart -- the same config file produced both when the
            # target was edited. data_teacher._verify_latents refuses a mismatch.
            "dataset_target": data_cfg["target"],
            "n": n,
            "paths_sha256": paths_fingerprint(paths),
            "shape": [n, zc, lat, lat],
            "dtype": "float32",
            "ckpt_path": str(cfg.model.params.get("ckpt_path", "")),
            "config": os.path.abspath(args.config),
            "gpu": torch.cuda.get_device_name(0),
        }, f, indent=2)

    print(f"wrote {args.out} ({n} rows) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
