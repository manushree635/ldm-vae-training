from __future__ import annotations
"""Epoch-pinned checkpoints that save_top_k cannot prune.

WHY THIS EXISTS.

main.py builds a ModelCheckpoint with monitor=model.monitor and save_top_k=3
(latent-diffusion/main.py:578-581). For the autoencoder that monitor is
val/rec_loss -- which is ANTI-CORRELATED with rFID on this data. So the default
callback keeps the three epochs with the lowest reconstruction loss and deletes
exactly the late epochs the rFID trajectory needs.

That is not hypothetical. Two runs lost their tails to it:

    steer  trained 25 epochs, scored trajectory ends at epoch 16
    f2c    trained 25 epochs, scored trajectory ends at epoch 19

and f2c's final checkpoint, recovered later from last.ckpt, scored 9.889 against
the 10.090 its last surviving epoch showed -- a 0.20 rFID difference that read as
"the schedule lost" when it was really "we stopped looking early".

last.ckpt survives pruning but holds only ONE epoch, so it cannot give a
trajectory. This callback writes to its own directory with no monitor and no
top-k, so nothing it saves is ever deleted.

LAYOUT, mirroring distillation_nerc's PeriodicSavePretrainedCallback:

    logs/<run>/checkpoints_periodic/
        checkpoint_epoch_0001.ckpt
        checkpoint_epoch_0005.ckpt
        ...

Epoch numbers are 1-INDEXED and zero-padded to four digits, matching nerc's
checkpoint_epoch_XXXX/ folders so the two trees read the same way. (nerc saves
HuggingFace directories via save_pretrained; that is not available here because
venv_ldm has no diffusers -- the conversion happens later, out of process, in
venv_train. So these are Lightning .ckpt files.)

DISK. Each full checkpoint is ~1.03 GB, so 25 epochs is ~26 GB per run. Three
ways to bound it, in order of preference:

  * `epochs: [1, 5, 10, 15, 20, 25]` -- save only what you will actually score.
  * `max_keep: 6` -- ring buffer, oldest periodic checkpoint deleted first.
    Never touches last.ckpt or the main ModelCheckpoint's files.
  * `save_weights_only: true` -- roughly halves each file by dropping optimizer
    state. Enough to score or export from; NOT enough to resume or post-tune,
    since Adam moments and global_step go with it. Default is False.
"""

import os

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


class PeriodicCheckpoint(Callback):
    """Save a full checkpoint at chosen epochs, outside save_top_k's reach.

    Args:
        dirpath: where to write. Relative paths resolve under the run's logdir,
            so the default keeps everything inside logs/<run>/.
        every_n_epochs: save when (epoch+1) % n == 0. 0 disables the cadence.
        epochs: explicit 1-indexed epochs to save, e.g. [1, 5, 25]. Combines
            with every_n_epochs by union -- either rule firing saves.
        max_keep: keep at most this many periodic checkpoints, deleting the
            oldest first. 0 (default) keeps everything.
        save_weights_only: drop optimizer state. Halves size, forfeits resume.
        save_on_train_end: always save the final epoch, whatever the rules say.
            On by default -- the final epoch is the one most often wanted and
            most often missed.
    """

    def __init__(self, dirpath="checkpoints_periodic", every_n_epochs=0, epochs=None,
                 max_keep=0, save_weights_only=False, save_on_train_end=True):
        super().__init__()
        self.dirpath = str(dirpath)
        self.every_n_epochs = int(every_n_epochs)
        self.epochs = set(int(e) for e in (epochs or []))
        self.max_keep = int(max_keep)
        self.save_weights_only = bool(save_weights_only)
        self.save_on_train_end = bool(save_on_train_end)
        self._written = []                  # in save order, for max_keep
        if self.every_n_epochs <= 0 and not self.epochs and not self.save_on_train_end:
            raise ValueError("PeriodicCheckpoint would never fire: set every_n_epochs, "
                             "epochs, or leave save_on_train_end on")

    def _resolve_dir(self, trainer):
        # trainer.logdir is set by main.py right after the Trainer is built.
        if os.path.isabs(self.dirpath):
            return self.dirpath
        return os.path.join(getattr(trainer, "logdir", "."), self.dirpath)

    def _should_save(self, epoch_1idx):
        if epoch_1idx in self.epochs:
            return True
        return self.every_n_epochs > 0 and epoch_1idx % self.every_n_epochs == 0

    @rank_zero_only
    def _save(self, trainer, epoch_1idx, why):
        out_dir = self._resolve_dir(trainer)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "checkpoint_epoch_{:04d}.ckpt".format(epoch_1idx))
        if os.path.exists(path):
            return
        trainer.save_checkpoint(path, weights_only=self.save_weights_only)
        self._written.append(path)
        size_gb = os.path.getsize(path) / 1e9 if os.path.exists(path) else 0.0
        print("[periodic-ckpt] epoch {:04d} -> {} ({:.2f} GB, {})".format(
            epoch_1idx, path, size_gb, why))
        self._prune()

    def _prune(self):
        if self.max_keep <= 0:
            return
        while len(self._written) > self.max_keep:
            old = self._written.pop(0)
            try:
                os.remove(old)
                print("[periodic-ckpt] pruned {} (max_keep={})".format(old, self.max_keep))
            except OSError as e:
                print("[periodic-ckpt] could not prune {}: {}".format(old, e))

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_1idx = int(trainer.current_epoch) + 1
        self._last_epoch_1idx = epoch_1idx
        if self._should_save(epoch_1idx):
            self._save(trainer, epoch_1idx, "scheduled")

    def on_train_end(self, trainer, pl_module):
        if not self.save_on_train_end:
            return
        # Name the final save by the last epoch the epoch hook actually saw. The first
        # version assumed trainer.current_epoch had already advanced here; on PL 1.4.2 it
        # has NOT, so a 25-epoch run wrote its final state a second time, mislabelled
        # checkpoint_epoch_0024.ckpt (same global_step as 0025). Tracking the epoch hook
        # is version-independent, and _save skips an existing file, so a final epoch the
        # cadence already saved is not written twice.
        epoch_1idx = getattr(self, "_last_epoch_1idx", None)
        if epoch_1idx is None:                      # e.g. resumed and ended at once
            epoch_1idx = int(trainer.current_epoch) + 1
        self._save(trainer, epoch_1idx, "final")
