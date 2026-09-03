from __future__ import annotations
"""Weights & Biases logging for the LDM trainer.

main.py already knows how to build a WandbLogger -- `default_logger_cfgs` has a
"wandb" entry -- but two things stop images from arriving:

  * `ImageLogger.logger_log_images` maps ONLY `pl.loggers.TestTubeLogger` to a
    handler. With a WandbLogger the lookup misses and falls back to a no-op, so
    reconstructions are written to logs/<run>/images/ on disk and nowhere else.
    `ImageLoggerWandb` adds the missing handler.

  * upstream logs whatever batch happens to be in flight, so the panel shows a
    different image every time and nothing can be compared across steps, epochs
    or runs. `FixedImageReconLogger` reconstructs ONE pinned image on a fixed
    cadence, which is what makes a wandb panel readable as a time series.

Both are opt-in from the config; nothing changes unless the config asks for them.
"""

import os

import numpy as np
import torch
import torchvision
import pytorch_lightning as pl
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from main import ImageLogger


def _to_uint8_hwc(grid: torch.Tensor, rescale: bool = True) -> np.ndarray:
    """CHW float in [-1,1] (or [0,1]) -> HWC uint8, matching upstream's log_local."""
    if rescale:
        grid = (grid + 1.0) / 2.0
    grid = grid.clamp(0.0, 1.0).detach().cpu()
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


class ImageLoggerWandb(ImageLogger):
    """Upstream's ImageLogger with a WandbLogger handler registered.

    Everything else -- the batch_frequency schedule, max_images, the local PNG
    dump, the train/val split handling -- is inherited untouched, so switching
    to this class changes where images go and nothing about when or which.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger_log_images[pl.loggers.WandbLogger] = self._wandb

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        import wandb
        payload = {}
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            payload[f"{split}/{k}"] = wandb.Image(_to_uint8_hwc(grid, self.rescale))
        pl_module.logger.experiment.log(payload, step=pl_module.global_step)


class FixedImageReconLogger(Callback):
    """Reconstruct ONE pinned image on a fixed cadence and log it to wandb.

    The preprocessing deliberately mirrors ldm.data.imagenet.ImageNetSR with
    min_crop_f = max_crop_f = 1.0 and random_crop = False -- centre-crop to the
    short side, resize to `size`, then x/127.5 - 1 -- so the tensor the model
    sees here is the same kind of tensor it sees in training.

    If the module exposes `ema_scope` (the EMA variants do) a second
    reconstruction is logged under the EMA weights, which is the direct read on
    whether the shadow copy is ahead of the live one.
    """

    def __init__(self, image_path, every_n_steps=500, size=256,
                 log_key="fixed", log_source_once=True):
        super().__init__()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"fixed image not found: {image_path}")
        self.image_path = image_path
        self.every_n_steps = int(every_n_steps)
        self.size = int(size)
        self.log_key = log_key
        self.log_source_once = log_source_once
        self._source_logged = False
        self._x = None                      # lazily built, cached

    def _load(self, device):
        if self._x is not None:
            return self._x.to(device)
        im = Image.open(self.image_path).convert("RGB")
        s = min(im.size)                    # centre-crop to the short side
        left, upper = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((left, upper, left + s, upper + s))
        im = im.resize((self.size, self.size), Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        self._x = torch.from_numpy(arr).permute(2, 0, 1)[None]
        return self._x.to(device)

    @rank_zero_only
    def _log(self, pl_module):
        import wandb
        if not isinstance(pl_module.logger, pl.loggers.WandbLogger):
            return
        x = self._load(pl_module.device)
        was_training = pl_module.training
        pl_module.eval()
        payload = {}
        try:
            with torch.no_grad():
                rec, _ = pl_module(x)
                payload[f"{self.log_key}/reconstruction"] = wandb.Image(_to_uint8_hwc(rec[0]))
                payload[f"{self.log_key}/psnr"] = float(
                    10.0 * torch.log10(4.0 / torch.mean((rec - x) ** 2).clamp_min(1e-12)))
                if hasattr(pl_module, "ema_scope") and getattr(pl_module, "use_ema", False):
                    with pl_module.ema_scope():
                        rec_ema, _ = pl_module(x)
                    payload[f"{self.log_key}/reconstruction_ema"] = wandb.Image(_to_uint8_hwc(rec_ema[0]))
                    payload[f"{self.log_key}/psnr_ema"] = float(
                        10.0 * torch.log10(4.0 / torch.mean((rec_ema - x) ** 2).clamp_min(1e-12)))
            if self.log_source_once and not self._source_logged:
                payload[f"{self.log_key}/source"] = wandb.Image(_to_uint8_hwc(x[0]))
                self._source_logged = True
            pl_module.logger.experiment.log(payload, step=pl_module.global_step)
        finally:
            if was_training:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = pl_module.global_step
        if step % self.every_n_steps == 0:
            self._log(pl_module)

    def on_train_start(self, trainer, pl_module):
        self._log(pl_module)                # step 0 baseline, before any update
