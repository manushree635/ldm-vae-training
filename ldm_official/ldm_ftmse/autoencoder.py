"""Decoder-only AutoencoderKL, matching Stability's sd-vae-ft-* finetunes."""

from contextlib import contextmanager

import torch

from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.ema import LitEma


class AutoencoderKLDecoderOnly(AutoencoderKL):
    """LDM's AutoencoderKL restricted to the parameters ft-MSE actually moved.

    Stability describe sd-vae-ft-ema / sd-vae-ft-mse as decoder finetunes, and
    the checkpoints bear that out. Diffing SD-1.5's VAE against both, per
    submodule (max |delta| over every tensor):

        encoder           0.000000   (bit-identical)
        quant_conv        0.000000   (bit-identical)
        decoder           1.293646
        post_quant_conv   0.057824

    So the trainable set is decoder + post_quant_conv, and the encoder half of
    the graph is frozen outright rather than merely excluded from the optimiser
    -- that keeps autograd from building a backward graph through it, which is
    also where the activation-memory saving comes from.

    Everything else -- training_step, validation_step, log_images, the KL term,
    get_last_layer -- is inherited from upstream untouched.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for module in (self.encoder, self.quant_conv):
            module.requires_grad_(False)

    def init_from_ckpt(self, path, ignore_keys=list()):
        """Upstream's loader, but loud about what did and did not land.

        Upstream calls `load_state_dict(..., strict=False)` and discards the
        result, so a checkpoint whose keys do not match silently leaves the
        model at its random init and training proceeds looking healthy. For a
        finetune that failure mode is invisible until the run is wasted, so
        assert the weights actually arrived.
        """
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)

        # loss.* (LPIPS + the discriminator) is built fresh and is expected to
        # be missing from a bare autoencoder checkpoint; nothing else may be.
        core_missing = [k for k in missing if not k.startswith("loss.")]
        print(f"Restored from {path}")
        print(f"  loaded {len(sd)} tensors; "
              f"{len(missing)} missing ({len(core_missing)} outside loss.*), "
              f"{len(unexpected)} unexpected")
        if unexpected:
            print(f"  unexpected: {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
        if core_missing:
            raise RuntimeError(
                f"{len(core_missing)} autoencoder tensors were not present in {path}, "
                f"e.g. {core_missing[:8]}. Refusing to start a finetune from a "
                f"partially random init."
            )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.decoder.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[decoder-only] optimising {n_train/1e6:.2f}M of {n_total/1e6:.2f}M "
              f"autoencoder+loss parameters at lr={lr}")
        return [opt_ae, opt_disc], []


class AutoencoderKLDecoderOnlyEMA(AutoencoderKLDecoderOnly):
    """Decoder-only AutoencoderKL with the EMA that upstream only gave VQModel.

    Stability's released sd-vae-ft-ema / sd-vae-ft-mse both "use EMA weights",
    but LDM's AutoencoderKL has no EMA at all -- `use_ema`, `LitEma`,
    `ema_scope` and the `on_train_batch_end` hook that actually updates the
    shadow weights all live in VQModel (autoencoder.py lines 14-263), and
    AutoencoderKL (line 285+) inherits none of it. This ports those pieces
    verbatim, changing only what has to change.

    Two details that matter:

    * LitEma is built AFTER the encoder is frozen, because it only shadows
      parameters with requires_grad -- constructing it before the freeze would
      keep a pointless shadow of the entire encoder.
    * LitEma's decay warms up as min(0.9999, (1+n)/(10+n)), so the averaging
      window grows with training instead of sitting at 0.9999. Over a few
      thousand steps that is ~0.991 at 1k and ~0.998 at 4k -- which is why a
      short run does not need the decay lowered by hand.

    Validation is reported twice, raw and under `ema_scope`, so `val/rec_loss`
    and `val_ema/rec_loss` give a direct read on whether the EMA is helping.
    """

    def __init__(self, *args, use_ema=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)     # after the freeze -- see docstring
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def validation_step(self, batch, batch_idx):
        out = super().validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                inputs = self.get_input(batch, self.image_key)
                reconstructions, posterior = self(inputs)
                _, log_ema = self.loss(inputs, reconstructions, posterior, 0,
                                       self.global_step,
                                       last_layer=self.get_last_layer(), split="val_ema")
                self.log("val_ema/rec_loss", log_ema["val_ema/rec_loss"])
                self.log_dict(log_ema)
        return out
