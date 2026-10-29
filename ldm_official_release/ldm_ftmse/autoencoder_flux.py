from __future__ import annotations
"""Finetune the DIFFUSERS VAE itself (FLUX.1's 16-channel kl-f8) inside the ldm training loop.

Why a separate class rather than converting the weights. FLUX's VAE has no quant_conv/post_quant_conv and a
16-channel latent, so putting it into ldm's AutoencoderKL means inserting identity 1x1 convs and re-keying
every tensor -- and the round trip did not reproduce diffusers bit-for-bit (latent 8.6e-5, decode 6.4e-4,
against 5.8e-5 for the 4-channel conversion). This class instead HOLDS the diffusers module and calls it, so
the model being finetuned IS the published one: nothing is converted, and there is nothing to verify.

What it keeps from the ldm side, so every existing loss/callback/config works unchanged:

  * the LightningModule interface main.py drives: get_input / training_step / validation_step /
    configure_optimizers / get_last_layer / log_images, with the same two-optimizer (AE, critic) protocol;
  * the loss is built from `lossconfig`, so ldm_ftmse.losses_steerable and every subclass apply as-is. They
    receive (inputs, reconstructions, posterior, optimizer_idx, global_step, last_layer, split);
  * the teacher stash of AutoencoderKLSteerableEMA (`loss.teacher_input`, plus `loss.current_epoch`), so the
    steerable terms and their band schedules behave identically;
  * decoder-only training with LitEma over the trainable parameters, built AFTER the freeze;
  * `kl_weight` in the loss still sees a real posterior: diffusers' DiagonalGaussianDistribution exposes
    .sample()/.mode()/.kl() with the same signatures ldm's loss calls.

Differences from the 4-channel arms, all forced by FLUX itself:
  * latent is 16 x H/8 x W/8, and the loss's kl term is over 16 channels;
  * there is no post_quant_conv to train, so opt_ae is the decoder alone;
  * FLUX's config carries scaling_factor 0.3611 / shift_factor 0.1159. They are NOT applied here: this is a
    pure autoencoder finetune whose losses live in image space, exactly as the 4-channel arms.

    model:
      target: ldm_ftmse.autoencoder_flux.AutoencoderKLFluxDecoderOnlyEMA
      params:
        vae_repo: black-forest-labs/FLUX.1-schnell
        vae_subfolder: vae
        use_ema: True
        teacher_key: teacher
        lossconfig: {...}
"""
import contextlib

import pytorch_lightning as pl
import torch

from ldm.modules.ema import LitEma
from ldm.util import instantiate_from_config


class AutoencoderKLFluxDecoderOnlyEMA(pl.LightningModule):
    def __init__(self,
                 lossconfig,
                 vae_repo="black-forest-labs/FLUX.1-schnell",
                 vae_subfolder="vae",
                 vae_class="AutoencoderKL",   # any diffusers autoencoder with .encoder/.decoder and
                                              # encode().latent_dist / decode().sample, e.g. AutoencoderKLFlux2
                 image_key="image",
                 teacher_key="teacher",
                 use_ema=True,
                 adam_betas=(0.5, 0.9),
                 monitor=None,
                 embed_dim=None,          # accepted and ignored: FLUX fixes it (16)
                 ddconfig=None,           # accepted and ignored: the architecture comes from the repo
                 ckpt_path=None,          # optional: resume decoder weights from a previous run of THIS class
                 ignore_keys=(),
                 **unused):
        super().__init__()
        import diffusers as _diffusers
        DiffusersAutoencoderKL = getattr(_diffusers, str(vae_class))   # AutoencoderKL unless a config says otherwise

        self.image_key = image_key
        self.teacher_key = teacher_key
        self.adam_betas = tuple(adam_betas) if adam_betas is not None else None
        self.vae = DiffusersAutoencoderKL.from_pretrained(vae_repo, subfolder=vae_subfolder,
                                                          torch_dtype=torch.float32)
        self.latent_channels = int(self.vae.config.latent_channels)
        self.loss = instantiate_from_config(lossconfig)
        if monitor is not None:
            self.monitor = monitor

        # --- decoder-only: freeze the encoder (and quant_conv if this VAE has one) ---------------
        frozen = 0
        for name, module in (("encoder", self.vae.encoder), ("quant_conv", getattr(self.vae, "quant_conv", None))):
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
                frozen += p.numel()
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("[flux] {} ({}) z={} | trainable {:.1f}M | frozen encoder {:.1f}M".format(
            vae_repo, vae_class, self.latent_channels, trainable / 1e6, frozen / 1e6))

        if ckpt_path is not None:
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd = sd.get("state_dict", sd)
            for k in list(sd):
                if any(k.startswith(ig) for ig in ignore_keys):
                    del sd[k]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            core = [k for k in missing if not k.startswith(("loss.", "model_ema."))]
            if core:
                raise RuntimeError("ckpt_path is missing model tensors: {}".format(core[:6]))
            print("[flux] resumed from {} ({} unexpected)".format(ckpt_path, len(unexpected)))

        # LitEma AFTER the freeze: it only shadows parameters with requires_grad.
        self.use_ema = bool(use_ema)
        if self.use_ema:
            self.model_ema = LitEma(self)
            print("[flux] LitEma over {} tensors".format(len(list(self.model_ema.buffers()))))

    # ---------------------------------------------------------------- ldm interface
    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        return x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

    def encode(self, x):
        return self.vae.encode(x).latent_dist

    def decode(self, z):
        return self.vae.decode(z).sample

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior

    def get_last_layer(self):
        return self.vae.decoder.conv_out.weight

    def train(self, mode=True):
        super().train(mode)
        self.vae.encoder.eval()          # frozen: keep it out of train mode
        return self

    # ---------------------------------------------------------------- EMA
    @contextlib.contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    # ---------------------------------------------------------------- steps
    def _stash_teacher(self, batch):
        if self.teacher_key in batch:
            self.loss.teacher_input = self.get_input(batch, self.teacher_key).detach()
        if hasattr(self.loss, "current_epoch"):
            self.loss.current_epoch = int(self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self._stash_teacher(batch)
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        if optimizer_idx == 0:
            aeloss, log_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                       last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss
        discloss, log_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                       last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return discloss

    def validation_step(self, batch, batch_idx):
        self._stash_teacher(batch)
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        _, log_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                              last_layer=self.get_last_layer(), split="val")
        _, log_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                last_layer=self.get_last_layer(), split="val")
        self.log("val/rec_loss", log_ae["val/rec_loss"])
        self.log_dict(log_ae)
        self.log_dict(log_disc)
        if self.use_ema:
            with self.ema_scope():
                rec_ema, post_ema = self(inputs)
                _, log_ema = self.loss(inputs, rec_ema, post_ema, 0, self.global_step,
                                       last_layer=self.get_last_layer(), split="val_ema")
                self.log_dict(log_ema)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = [p for p in self.vae.decoder.parameters() if p.requires_grad]
        pq = getattr(self.vae, "post_quant_conv", None)
        if pq is not None:
            params += [p for p in pq.parameters() if p.requires_grad]
        kw = {} if self.adam_betas is None else {"betas": self.adam_betas}
        opt_ae = torch.optim.Adam(params, lr=lr, **kw)
        disc = getattr(self.loss, "discriminator", None)
        if disc is None or float(getattr(self.loss, "disc_factor", 1.0)) == 0.0:
            print("[flux] single optimiser (no GAN): {:.1f}M decoder params at lr={}".format(
                sum(p.numel() for p in params) / 1e6, lr))
            return opt_ae
        opt_disc = torch.optim.Adam(disc.parameters(), lr=lr, **kw)
        return [opt_ae, opt_disc], []

    # ---------------------------------------------------------------- image logging
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = {}
        x = self.get_input(batch, self.image_key).to(self.device)
        if not only_inputs:
            xrec, _ = self(x)
            log["reconstructions"] = xrec
            if self.use_ema:
                with self.ema_scope():
                    log["reconstructions_ema"] = self(x)[0]
        log["inputs"] = x
        if self.teacher_key in batch:
            log["teacher"] = self.get_input(batch, self.teacher_key).to(self.device)
        return log
