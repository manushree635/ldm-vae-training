"""
sd-vae-ft-* overlay for CompVis/latent-diffusion.

The three upstream clones (latent-diffusion, taming-transformers, CLIP) are kept
byte-identical to their upstream HEADs. Everything Stability's sd-vae-ft-mse
recipe needs that stock LDM cannot express lives here, as thin subclasses that
the training YAML points at via `target:`.

Two deltas, and only two:

  1. AutoencoderKLDecoderOnly -- upstream optimises encoder + decoder +
     quant_conv + post_quant_conv. Both ft-EMA and ft-MSE are decoder-only
     finetunes. Used by both configs.
  2. LPIPSWithDiscriminatorMSE -- upstream's LPIPSWithDiscriminator hardcodes an
     L1 reconstruction term, which is already the ft-EMA recipe, so the ft-EMA
     config does not use this class at all and points straight at upstream's.
     Only ft-MSE ("MSE + 0.1 * LPIPS") needs it.

The decoder-only trainable set was not guessed: diffing SD-1.5's VAE against
stabilityai/sd-vae-ft-mse shows encoder and quant_conv are bit-identical
(max |delta| == 0.0), while decoder (1.29) and post_quant_conv (0.058) moved.
"""
