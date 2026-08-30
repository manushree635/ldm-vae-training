"""Graft LDM's pretrained discriminator onto the SD-1.5 VAE weights.

CompVis' released kl-f8 (ommer-lab kl-f8.zip) is a full PyTorch-Lightning
TRAINING checkpoint, not an inference export, so it still carries the
LPIPSWithDiscriminator state -- 22 `loss.discriminator.*` tensors trained for
246,803 steps. That matters because it is what makes an early `disc_start`
sane: a randomly-initialised NLayerDiscriminator produces meaningless gradients
for thousands of steps, which is the reason upstream defaults to disc_start=50001.

Caveat worth keeping in mind: kl-f8 is a DIFFERENT autoencoder from SD-1.5's VAE
(same architecture, trained by CompVis on OpenImages). The discriminator has
learned real-vs-reconstructed features for f8 decoders in general; it is not
tuned to this particular decoder.

    python scripts/merge_disc_into_ckpt.py --vae models/sd15_vae_ldm.ckpt \
        --disc <kl-f8>/model.ckpt --out models/sd15_vae_ldm_with_disc.ckpt
"""
import argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True, help="LDM-layout VAE checkpoint")
    ap.add_argument("--disc", required=True, help="kl-f8 training checkpoint")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vae = torch.load(args.vae, map_location="cpu", weights_only=False)
    vae_sd = vae["state_dict"] if "state_dict" in vae else vae
    disc_ck = torch.load(args.disc, map_location="cpu", weights_only=False)
    disc_sd = disc_ck["state_dict"]

    disc = {k: v for k, v in disc_sd.items() if k.startswith("loss.discriminator")}
    if not disc:
        raise SystemExit(f"no loss.discriminator.* tensors in {args.disc}")
    overlap = set(vae_sd) & set(disc)
    if overlap:
        raise SystemExit(f"unexpected key collision: {sorted(overlap)[:5]}")

    merged = dict(vae_sd); merged.update(disc)
    torch.save({"state_dict": merged}, args.out)
    print(f"vae tensors        : {len(vae_sd)}")
    print(f"discriminator      : {len(disc)}  (from step {disc_ck.get('global_step')})")
    print(f"merged             : {len(merged)} -> {args.out}")
    # LPIPS is rebuilt from taming's vgg.pth at init, so it is deliberately not merged.
    print("note: loss.perceptual_loss.* intentionally omitted (rebuilt from vgg.pth)")


if __name__ == "__main__":
    main()
