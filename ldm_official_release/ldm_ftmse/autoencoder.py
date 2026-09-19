"""Decoder-only AutoencoderKL, matching Stability's sd-vae-ft-* finetunes."""

from contextlib import contextmanager

import torch

from ldm.models.autoencoder import AutoencoderKL
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.modules.ema import LitEma

#: Batch key holding precomputed `quant_conv(encoder(x))` moments. Written by
#: scripts/precompute_latents.py, attached by ldm_ftmse.data_teacher when its
#: `latents_file` param is set. Absent -> the encoder runs as before.
LATENTS_KEY = "moments"


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

    log_images, the KL term and get_last_layer are inherited from upstream
    untouched. training_step and validation_step are carried rather than
    inherited, for two changes that are BIT-EXACT by construction -- see
    `_ae_forward` and `training_step`.
    """

    def __init__(self, *args, disc_reuse_recon=False, channels_last=False,
                 allow_tf32=False, **kwargs):
        super().__init__(*args, **kwargs)
        for module in (self.encoder, self.quant_conv):
            module.requires_grad_(False)

        # --- opt-in speed flags. All default OFF, so an existing config is
        # --- untouched. Only the latent cache and disc_skip_recon_loss are
        # --- bit-exact; the two below are NOT -- see each one.
        self.disc_reuse_recon = bool(disc_reuse_recon)
        self._recon_cache = None           # plain attribute, never a buffer
        self.use_channels_last = bool(channels_last)

        if self.disc_reuse_recon and not getattr(self.loss, "disc_skip_recon_loss", False):
            raise ValueError(
                "disc_reuse_recon=True requires lossconfig.params.disc_skip_recon_loss"
                "=True. Reusing the generator's reconstruction means no posterior is "
                "computed for the discriminator step, and upstream's loss calls "
                "posteriors.kl() there unconditionally (contperceptual.py:59).")

        if allow_tf32:
            # torch 1.11 defaults matmul TF32 OFF while cudnn TF32 is ON, so the
            # convolutions already use it but the mid-block attention bmms run in
            # true fp32. NOT bit-exact.
            torch.backends.cuda.matmul.allow_tf32 = True
        if self.use_channels_last:
            self.to(memory_format=torch.channels_last)
        if self.disc_reuse_recon or self.use_channels_last or allow_tf32:
            print(f"[fast] disc_reuse_recon={self.disc_reuse_recon} "
                  f"channels_last={self.use_channels_last} allow_tf32={bool(allow_tf32)} "
                  f"disc_skip_recon_loss={getattr(self.loss, 'disc_skip_recon_loss', False)}")

    def get_input(self, batch, k):
        x = super().get_input(batch, k)
        if self.use_channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

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
        ae_groups = [{"params": list(self.decoder.parameters()) +
                                list(self.post_quant_conv.parameters())}]
        # Loss-side learned upsampler (losses_steerable.py `steerable_upsampler:
        # subpixel`). It is judged with the generator, so it trains in opt_ae, in
        # its own group so its lr can be scaled. Absent -> one group, as before.
        up_head = getattr(self.loss, "up_head", None)
        head_note = ""
        if up_head is not None:
            mult = float(getattr(self.loss, "steerable_upsampler_lr_mult", 1.0))
            ae_groups.append({"params": list(up_head.parameters()), "lr": lr * mult})
            n_head = sum(p.numel() for p in up_head.parameters())
            head_note = f" (+ {n_head} sub-pixel head params at lr={lr * mult:.3g})"
        opt_ae = torch.optim.Adam(ae_groups, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[decoder-only] optimising {n_train/1e6:.2f}M of {n_total/1e6:.2f}M "
              f"autoencoder+loss parameters at lr={lr}{head_note}")
        return [opt_ae, opt_disc], []

    # ------------------------------------------------------------------ #
    # Bit-exact step-time savings.  Both rest on facts specific to THIS   #
    # setup, so both are opt-in on the data and inert without it.         #
    # ------------------------------------------------------------------ #

    def _ae_forward(self, batch, inputs):
        """`self(inputs)`, but skipping the encoder when its output is cached.

        The encoder and quant_conv are frozen (see __init__), and the teacher
        datasets force `min_crop_f = max_crop_f = 1.0, random_crop = False`
        (data_teacher.py) -- so `quant_conv(encoder(x_i))` is the same tensor on
        every epoch, forever, and upstream recomputes it TWICE per batch (once
        per optimizer index). Precomputing it once is therefore not an
        approximation, it is the removal of repeated work.

        Bit-exactness turns on `sample()` drawing exactly one `torch.randn` of
        the same shape either way, which keeps the RNG stream aligned:

            x = self.mean + self.std * torch.randn(self.mean.shape)   # CPU RNG

        Without `moments` in the batch this is upstream's forward verbatim.
        """
        moments = batch.get(LATENTS_KEY) if isinstance(batch, dict) else None
        if moments is None:
            return self(inputs)
        # .float() mirrors get_input(); the cache is written fp32 precisely so
        # this is a no-op cast rather than a precision change.
        posterior = DiagonalGaussianDistribution(
            moments.to(device=inputs.device).float())
        z = posterior.sample()
        if self.use_channels_last:
            z = z.to(memory_format=torch.channels_last)
        return self.decode(z), posterior

    def training_step(self, batch, batch_idx, optimizer_idx):
        """Upstream's training_step with the discriminator branch's forward
        placed under `no_grad`.

        Upstream runs `self(inputs)` unconditionally at the top, for both
        optimizer indices. The `optimizer_idx == 1` branch of the loss consumes
        only `reconstructions.contiguous().detach()` and never backpropagates
        into the autoencoder (contperceptual.py:94-110), so the graph that
        second forward builds is constructed and then discarded.

        `no_grad` changes no kernel -- only whether the graph is recorded -- so
        the returned d_loss and its log dict are bit-identical. The saving is
        mostly memory: it stops retaining a whole autoencoder forward's
        activations.

        NOT changed, deliberately: the loss still recomputes rec_loss, the LPIPS
        VGG forward and posteriors.kl() in the disc branch and throws them away
        (contperceptual.py:48-60). Removing that needs a carried copy of
        upstream's loss forward, which is a separate change with its own parity
        test -- see scripts/test_stepcache_parity.py's docstring.
        """
        inputs = self.get_input(batch, self.image_key)

        if optimizer_idx == 0:
            reconstructions, posterior = self._ae_forward(batch, inputs)
            if self.disc_reuse_recon:
                self._recon_cache = reconstructions.detach()
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx,
                                            self.global_step, last_layer=self.get_last_layer(),
                                            split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            if self.disc_reuse_recon:
                # NOT bit-exact, and deliberately so: Lightning steps opt_ae
                # between the two closures, so on accumulation boundaries the
                # discriminator now sees a PRE-update reconstruction. It also
                # removes one posterior.sample(), shifting the RNG stream. A
                # standard GAN formulation, but a method change -- which is why
                # it is off by default and never enabled on an in-flight arm.
                if self._recon_cache is None:
                    raise RuntimeError(
                        "disc_reuse_recon: no reconstruction stashed. The "
                        "optimizer_idx==0 closure must run first for each batch.")
                reconstructions, posterior = self._recon_cache, None
                self._recon_cache = None
            else:
                with torch.no_grad():
                    reconstructions, posterior = self._ae_forward(batch, inputs)
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx,
                                                self.global_step, last_layer=self.get_last_layer(),
                                                split="train")
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        """Upstream's validation_step, routed through `_ae_forward`.

        No `no_grad` addition here: Lightning already runs validation under
        `no_grad`, which is what makes `calculate_adaptive_weight` raise and
        upstream fall back to `d_weight = 0` (contperceptual.py:76-78). That
        fallback is load-bearing and is left exactly as it is.
        """
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self._ae_forward(batch, inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict


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
                reconstructions, posterior = self._ae_forward(batch, inputs)
                _, log_ema = self.loss(inputs, reconstructions, posterior, 0,
                                       self.global_step,
                                       last_layer=self.get_last_layer(), split="val_ema")
                self.log("val_ema/rec_loss", log_ema["val_ema/rec_loss"])
                self.log_dict(log_ema)
        return out
