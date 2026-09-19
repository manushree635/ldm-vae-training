from __future__ import annotations
"""Make CompVis's main.py run under BOTH pytorch-lightning 1.4.2 and 1.9.5.

Why one file for two versions rather than a forked main.py: all 28 configs name
`main.DataModuleFromConfig` / `main.ImageLogger` as targets, and the in-flight
20k arms are mid-run on the 1.4.2 stack. A fork would either strand those
configs or require editing all of them.

Why 1.9.5 as the ceiling: it is the LAST release that supports multiple
optimizers under AUTOMATIC optimisation. This repo uses `optimizer_idx` in 32
places; PL 2.x removed that path, which would turn the two-optimizer GAN loop
into a manual-optimisation rewrite -- and a rewritten training loop cannot be
gated by a loss-parity test against the old one.

Everything here is a shim for an API that MOVED, not a behaviour change.
"""

import pytorch_lightning as pl

PL_VERSION = tuple(int(p) for p in pl.__version__.split(".")[:2])
PL_GE_16 = PL_VERSION >= (1, 6)
PL_GE_17 = PL_VERSION >= (1, 7)
PL_GE_18 = PL_VERSION >= (1, 8)

# rank_zero_* moved out of utilities.distributed in 1.7.
try:                                                    # PL >= 1.7
    from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
except ImportError:                                     # PL 1.4
    from pytorch_lightning.utilities import rank_zero_info
    from pytorch_lightning.utilities.distributed import rank_zero_only

__all__ = ["PL_VERSION", "PL_GE_16", "PL_GE_17", "PL_GE_18",
           "rank_zero_info", "rank_zero_only",
           "root_device", "reduce_scalar", "default_logger_cfg",
           "image_logger_handler_key", "apply_ddp_config"]


def root_device(trainer):
    """`trainer.root_gpu` (int) -> `trainer.strategy.root_device` (torch.device).

    Both are accepted by torch.cuda.{reset_peak_memory_stats,synchronize,
    max_memory_allocated}, so callers need no further branching.
    """
    rg = getattr(trainer, "root_gpu", None)
    if rg is not None:
        return rg
    return trainer.strategy.root_device


def reduce_scalar(trainer, value):
    """`trainer.training_type_plugin` -> `trainer.strategy` in PL 1.8."""
    plugin = getattr(trainer, "strategy", None)
    if plugin is None:
        plugin = trainer.training_type_plugin
    return plugin.reduce(value)


def default_logger_cfg(logdir, nowname):
    """TestTubeLogger was REMOVED in PL 1.7; CSVLogger is the closest drop-in.

    Returns upstream's testtube config on 1.4 so existing runs are unchanged.
    """
    if not PL_GE_17:
        return {"target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {"name": "testtube", "save_dir": logdir}}
    return {"target": "pytorch_lightning.loggers.CSVLogger",
            "params": {"name": "csv", "save_dir": logdir}}


def image_logger_handler_key():
    """The logger class ImageLogger maps to its `_testtube` handler.

    Returns None on PL >= 1.7, where TestTubeLogger no longer exists -- callers
    skip registering that handler rather than crashing on the attribute.
    """
    return getattr(pl.loggers, "TestTubeLogger", None)


def apply_ddp_config(trainer_config, gpuinfo):
    """Translate upstream's PL-1.4 DDP spelling into whatever this PL wants.

    PL 1.4:  accelerator="ddp",  gpus="0,1,"
    PL 1.9:  accelerator="gpu",  devices=2,  strategy="ddp"

    `gpus` and `devices` are mutually exclusive on 1.9, so `gpus` is removed
    there. Returns the GPU COUNT, because main.py otherwise recovers it by
    re-parsing `lightning_config.trainer.gpus`, which no longer exists.
    """
    ngpu = len([g for g in str(gpuinfo).strip(",").split(",") if g != ""])
    if not PL_GE_16:
        trainer_config["accelerator"] = "ddp"
        return ngpu
    trainer_config["accelerator"] = "gpu"
    trainer_config["devices"] = ngpu
    trainer_config["strategy"] = "ddp"
    trainer_config.pop("gpus", None)
    return ngpu
