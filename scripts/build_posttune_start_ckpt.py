"""Build a post-tune START checkpoint from a training checkpoint: EMA weights in, EMA/optimizer out.

    python scripts/build_posttune_start_ckpt.py --src <run>/checkpoints/epoch=000024.ckpt \
        --out models/<tag>ema_vae_ldm_with_disc.ckpt --expect-gs 129000

Same recipe as models/w5ep13ema_vae_ldm_with_disc.ckpt (w5post), written down this time:
  * refuses unless --src is a true epoch end (is_last_batch, and global_step == --expect-gs if given);
  * every shadowed parameter (LitEma names are the param name with the dots removed) is replaced
    by its EMA value -- i.e. the exact weights the EMA exports and every evaluation scored
    (decoder + post_quant_conv + loss.logvar for the no-GAN runs);
  * model_ema.* is dropped, and no optimizer state is written: LitEma is built after
    init_from_ckpt, so the post-tune's EMA starts from these weights and global_step from 0;
  * everything else (frozen encoder, quant_conv, loss.discriminator.*, LPIPS) is kept as is.
    For a no-GAN source the discriminator is the pretrained kl-f8 critic it was initialised
    with (the no-GAN class freezes it), so the post-tune GAN starts from that critic.
"""
import argparse, os, sys, torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--expect-gs", type=int, default=None)
a = ap.parse_args()

ck = torch.load(a.src, map_location="cpu", weights_only=False)
bp = ck["loops"]["fit_loop"]["epoch_loop.batch_progress"]
gs = ck["global_step"]
print(f"src {a.src}\n  epoch={ck['epoch']} gs={gs} batches={bp['current']['completed']} is_last_batch={bp.get('is_last_batch')}")
if not bp.get("is_last_batch") or (a.expect_gs is not None and gs != a.expect_gs):
    sys.exit(f"FATAL: not the expected full-epoch checkpoint (want gs={a.expect_gs}, is_last_batch=True)")

sd = ck["state_dict"]
params = [k for k in sd if not k.startswith("model_ema.")]
swapped = 0
out = {}
for k in params:
    s = "model_ema." + k.replace(".", "")
    if s in sd:
        out[k] = sd[s].clone(); swapped += 1
    else:
        out[k] = sd[k]
dec = [k for k in params if k.startswith(("decoder.", "post_quant_conv."))]
missing = [k for k in dec if "model_ema." + k.replace(".", "") not in sd]
if missing:
    sys.exit(f"FATAL: {len(missing)} decoder/post_quant_conv params have no EMA shadow, e.g. {missing[:3]}")
if not any(k.startswith("loss.discriminator.") for k in out):
    sys.exit("FATAL: no loss.discriminator.* in the source -- the GAN would start from random init")

os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
tmp = a.out + ".tmp"
torch.save({"state_dict": out, "global_step": 0, "source": {"ckpt": a.src, "global_step": gs, "epoch": ck["epoch"]}}, tmp)
os.replace(tmp, a.out)
print(f"  swapped {swapped} EMA tensors (decoder+post_quant_conv {len(dec)}), kept {len(out)} tensors, "
      f"discriminator {sum(k.startswith('loss.discriminator.') for k in out)}\n  -> {a.out}")
