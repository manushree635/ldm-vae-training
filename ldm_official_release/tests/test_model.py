"""End-to-end model check on CPU: for every model class the configs use, build it from
the first config that names it (random init, ckpt_path dropped), run one training step
per optimizer at the real 256/512 geometry, and check the loss is finite and gradients
land only where they should.

Slow (~15 s per class). Skip with LDM_SKIP_SLOW=1; limit with LDM_MODEL_TEST_ONLY=<substr>.
Needs the LPIPS/VGG weights the training run itself needs; a class whose config points
at a file that is not on this machine is skipped, not failed.
"""
import glob
import os
import unittest

from _env import ROOT, rel

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

#: params that name files a fresh clone may not have; dropped so the build is random-init.
PATH_PARAMS = ("ckpt_path", "disc_ckpt_path", "init_ckpt", "stage1_ckpt")


def first_config_per_model_class():
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        if os.path.basename(f).startswith("_"):
            continue
        target = OmegaConf.load(f).get("model", {}).get("target")
        if target and target not in out:
            out[target] = f
    return out


def build(path):
    cfg = OmegaConf.load(path)
    for k in PATH_PARAMS:
        cfg.model.params.pop(k, None)
    model = instantiate_from_config(cfg.model)
    model.learning_rate = float(cfg.model.base_learning_rate)
    # training_step logs through the Trainer; there is none here.
    model.log = lambda *a, **k: None
    model.log_dict = lambda *a, **k: None
    type(model).global_step = property(lambda self: 10 ** 6)     # past disc_start: GAN term live
    type(model).current_epoch = property(lambda self: 0)
    return cfg, model.train()


def as_list(opts):
    if isinstance(opts, tuple) and len(opts) == 2 and isinstance(opts[0], (list, tuple)):
        opts = opts[0]
    return list(opts) if isinstance(opts, (list, tuple)) else [opts]


@unittest.skipIf(os.environ.get("LDM_SKIP_SLOW"), "LDM_SKIP_SLOW set")
class TestModelTrainingStep(unittest.TestCase):
    def test_one_step_per_model_class(self):
        # cwd matters: taming resolves taming/modules/autoencoder/lpips/vgg.pth against it.
        os.chdir(ROOT)
        torch.manual_seed(0)
        only = os.environ.get("LDM_MODEL_TEST_ONLY", "")
        items = [(t, f) for t, f in first_config_per_model_class().items() if only in t]
        self.assertTrue(items, "no model classes found in configs/")
        for target, path in items:
            with self.subTest(model=target.rsplit(".", 1)[1], config=rel(path)):
                # A diffusers-held VAE (vae_repo) is a directory outside the repo, so a
                # fresh clone may not have it. from_pretrained raises OSError rather than
                # FileNotFoundError, so check here instead of widening the catch below and
                # swallowing real failures with it.
                repo = OmegaConf.load(path).get("model", {}).get("params", {}).get("vae_repo")
                if repo and os.sep in str(repo) and not os.path.isdir(str(repo)):
                    self.skipTest("needs a VAE not on this machine: {}".format(repo))
                try:
                    cfg, model = build(path)
                except FileNotFoundError as e:
                    self.skipTest("needs a file not on this machine: {}".format(e))
                except ImportError as e:
                    # setup_env.sh installs diffusers, but it is optional for the ldm-native
                    # classes, so a venv without it should skip this arm rather than fail it.
                    self.skipTest("needs a package not in this venv: {}".format(e))
                batch = {model.image_key: torch.rand(1, 256, 256, 3) * 2 - 1}
                teacher_key = getattr(model, "teacher_key", None)
                if teacher_key:
                    batch[teacher_key] = torch.rand(1, 512, 512, 3) * 2 - 1

                opts = as_list(model.configure_optimizers())
                self.assertIn(len(opts), (1, 2))
                for oi, opt in enumerate(opts):
                    model.zero_grad(set_to_none=True)
                    loss = model.training_step(batch, 0, oi)
                    self.assertTrue(torch.isfinite(loss), "optimizer {}: loss {}".format(oi, loss))
                    loss.backward()
                    owned = {id(p) for g in opt.param_groups for p in g["params"]}
                    with_grad = [p for p in model.parameters() if p.grad is not None]
                    self.assertTrue(with_grad, "optimizer {}: no gradient at all".format(oi))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in with_grad),
                                    "optimizer {}: non-finite gradient".format(oi))
                    got = sum(1 for p in with_grad if id(p) in owned and p.grad.abs().sum() > 0)
                    self.assertGreater(got, 0, "optimizer {} owns no parameter that moved".format(oi))
                    opt.step()

                frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
                if "FullEnc" not in target and "Probe" not in target:
                    # "encoder." for the ldm classes, "vae.encoder." for the ones that hold a
                    # diffusers module instead of converting it (autoencoder_flux).
                    self.assertTrue(any(n.startswith(("encoder.", "vae.encoder.")) for n in frozen),
                                    "decoder-only class left the encoder trainable")
                del model


if __name__ == "__main__":
    unittest.main()
