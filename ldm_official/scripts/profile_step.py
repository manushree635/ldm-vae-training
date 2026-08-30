"""Decompose LDM's per-step cost: which part makes it slow?

Times one training step at the run's real shape (batch 12, 256px) under variants
that isolate each suspected factor, so the slowdown is attributed by measurement
rather than by argument.
"""
import time, torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

CFG = "/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/configs/autoencoder_kl_32x32x4_ftema_20k.yaml"
B, RES, WARM, ITERS = 12, 256, 3, 10


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / ITERS


def main():
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    cfg = OmegaConf.load(CFG)
    cfg.model.params.pop("ckpt_path", None)          # random init is fine for timing
    m = instantiate_from_config(cfg.model).to(dev)
    m.learning_rate = 4.5e-6
    # training_step calls self.log()/self.log_dict(), which need a live Trainer.
    # Stub them: they cost nothing and are irrelevant to step timing.
    m.log = lambda *a, **k: None
    m.log_dict = lambda *a, **k: None
    type(m).global_step = property(lambda self: 0)
    (opt_ae, opt_disc), _ = m.configure_optimizers()
    x = torch.randn(B, 3, RES, RES, device=dev)
    batch = {"image": x.permute(0, 2, 3, 1).contiguous()}

    def gen_step(amp=False):
        opt_ae.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=amp):
            loss = m.training_step(batch, 0, 0)
        loss.backward(); opt_ae.step()

    def disc_step():
        opt_disc.zero_grad(set_to_none=True)
        loss = m.training_step(batch, 0, 1)
        loss.backward(); opt_disc.step()

    def full():
        gen_step(); disc_step()

    lp = m.loss.perceptual_loss
    def lpips_only():
        with torch.no_grad():
            lp(x, x)

    r = {
        "full step (gen + disc, fp32)     ": timed(full),
        "generator step only (fp32)       ": timed(lambda: gen_step(False)),
        "generator step only (fp16 amp)   ": timed(lambda: gen_step(True)),
        "  [ VGG16 LPIPS forward alone ]  ": timed(lpips_only),
    }
    base = r["full step (gen + disc, fp32)     "]
    print(f"\nbatch {B} @ {RES}px on {torch.cuda.get_device_name(0)}  "
          f"({ITERS} iters after {WARM} warmup)\n")
    for k, v in r.items():
        print(f"  {k} {v*1000:8.1f} ms   {base/v:5.2f}x faster than full"
              if v != base else f"  {k} {v*1000:8.1f} ms   (baseline)")
    steps = 396
    print(f"\nprojected epoch time @ {steps} steps/rank:")
    for k, v in r.items():
        if "LPIPS" in k:
            continue
        print(f"  {k} {v*steps/60:6.2f} min")


if __name__ == "__main__":
    main()
