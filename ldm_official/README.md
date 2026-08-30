# LDM VAE — official codebase, ft-MSE recipe, 20k LAION faces

A from-scratch baseline that trains the SD/kl-f8 autoencoder using **CompVis'
own [latent-diffusion](https://github.com/CompVis/latent-diffusion) trainer**,
under Stability's **sd-vae-ft-\*** recipes, on the same 20k images and the same
train/val split that `distillation_nerc`'s 20k configs use.

Two configs, both decoder-only from the pre-finetune SD-1.x kl-f8 VAE:

| config | pixel term | LPIPS weight | loss class |
|---|---|---|---|
| `..._ftema_20k.yaml` **(default)** | L1 | 1.0 | upstream's, unmodified |
| `..._ftmse_20k.yaml` | MSE | 0.1 | one-line subclass |

ft-EMA is the plainer of the two: Stability describe it as using "the same loss
configuration as the original checkpoint", which *is* what LDM ships, so that
config points straight at `ldm.modules.losses.LPIPSWithDiscriminator` and the
loss overlay drops out of the picture entirely.

Nothing here shares code with `distillation_nerc`. It is a separate tree with
its own pinned environment.

```
ldm_official/
  latent-diffusion/     upstream @ a506df5   (unmodified)
  taming-transformers/  upstream @ 3ba01b2   (unmodified)
  CLIP/                 upstream            (unmodified)
  ldm_ftmse/            the ft-MSE overlay — the only new model/loss code
  configs/              the training YAML
  scripts/              setup, conversion, launch
  filelists/subset_20k/ train.txt (19,000) / val.txt (1,000)
  models/               SD-1.5 VAE converted to LDM key names
  logs/                 run dirs (checkpoints, images, tensorboard)
```

## The ft-MSE mapping, and where each piece comes from

(The ft-EMA config is the same minus the last two rows of deltas:
upstream's loss verbatim, `perceptual_weight` left at its 1.0 default.)

Stability describe sd-vae-ft-mse as: resumed from ft-EMA, **280k further steps**,
**decoder-only**, loss with "more emphasis on MSE reconstruction
(MSE + 0.1 · LPIPS)", **batch size 192**, **lr 4.5e-6**, on LAION-Aesthetics +
LAION-Humans. Mapped onto LDM's `autoencoder_kl_32x32x4.yaml`:

| Knob | Upstream LDM | Here | Why |
|---|---|---|---|
| `ddconfig` | kl-f8 | **unchanged** | this *is* the SD VAE architecture |
| `base_learning_rate` | `4.5e-6` | **unchanged** | upstream LDM's value; the model cards publish **no** lr |
| `kl_weight` | `1e-6` | **unchanged** | |
| `disc_weight` | `0.5` | **unchanged** | |
| `disc_start` | `50001` | **unchanged** | see *The discriminator* below |
| `perceptual_weight` | `1.0` (default) | `0.1` | the "0.1 · LPIPS" in the recipe |
| pixel term | hardcoded L1 | MSE | L1 *is* the ft-**EMA** recipe; ft-MSE is squared error |
| trained params | encoder+decoder+both quant convs | decoder + `post_quant_conv` | ft-* are decoder finetunes |
| `accumulate_grad_batches` | `2` | **unchanged** | global batch 96, not Stability's 192 — see below |
| `max_epochs` | *(unset)* | `50` | matches the distillation runs' budget |
| data pipeline | `ImageNetSRTrain` | same class, `get_base()` repointed | upstream's `__getitem__` inherited verbatim |

### The trainable set was measured, not assumed

Diffing SD-1.5's VAE against both finetunes, max |Δ| per submodule:

| module | vs ft-MSE | vs ft-EMA |
|---|---|---|
| `encoder` | **0.000000** | **0.000000** |
| `quant_conv` | **0.000000** | **0.000000** |
| `decoder` | 1.293646 | 0.901528 |
| `post_quant_conv` | 0.057824 | 0.051707 |

Encoder and `quant_conv` are bit-identical, so `AutoencoderKLDecoderOnly`
freezes them outright (not merely excluded from the optimiser — freezing also
stops autograd building a graph through the encoder).

## Image loading and resizing

Upstream's autoencoder config loads through `ldm.data.imagenet.ImageNetSRTrain`,
whose `get_base()` reads a pickled ImageNet index and instantiates
`ImageNetTrain` (which auto-downloads ImageNet). There is no way to aim it at a
directory — but the transform logic is not in `get_base()`, it is all in
`ImageNetSR.__getitem__`. So `ldm_ftmse/data.py` subclasses `ImageNetSR` and
overrides **`get_base()` only**, yielding `{"file_path_": ...}` dicts from a
file list. The pipeline itself is inherited, not reimplemented:

    crop_side = min_side * U(min_crop_f=0.5, max_crop_f=1.0)
    RandomCrop(crop_side, crop_side)
    SmallestMaxSize(256, interpolation=cv2.INTER_AREA)
    x / 127.5 - 1.0

Checked on the instantiated dataset: `type(ds).__getitem__ is
ImageNetSR.__getitem__` → `True` (the same function object), `INTER_AREA`,
`center_crop=False`, and two draws of the same index differ, so the
augmentation is live. The `pil_nearest` `LR_image` branch runs too and is
ignored by `AutoencoderKL`, exactly as upstream.

Note `random_crop` defaults to `True` for **validation** as well, because
upstream's config passes only `size` and `degradation` to
`ImageNetSRValidation`. That is faithful, but it makes `val/rec_loss` noisy and
that metric drives `save_top_k=3`. Add `random_crop: False` to the validation
block for a stable monitor, at the cost of one line off upstream.

Upstream's `__getitem__` draws from `np.random`, and PyTorch does not reseed
numpy per worker, so the 4 workers share a stream. `main.py` ships
`worker_init_fn` for exactly this; upstream's autoencoder config does not enable
it, so neither does this one. `use_worker_init_fn: True` turns it on.

## What was actually modified

The three clones are byte-identical to upstream. Every delta lives in
`ldm_ftmse/`, which the YAML reaches through `target:`:

- **`ldm_ftmse/losses.py`** — `LPIPSWithDiscriminatorMSE`. Upstream's
  `forward` opens with a hardcoded
  `rec_loss = torch.abs(inputs - reconstructions)`. It is inlined at the top of
  a 60-line method, so overriding that one line means carrying the method; the
  copy is verbatim apart from the marked line. `rec_loss_type: l1` reproduces
  upstream bit-for-bit.
- **`ldm_ftmse/autoencoder.py`** — `AutoencoderKLDecoderOnly`. Freezes the
  encoder, narrows `configure_optimizers`, and makes `init_from_ckpt` loud:
  upstream calls `load_state_dict(strict=False)` and throws the result away, so
  a key mismatch silently leaves the model at random init and the run looks
  healthy until it is wasted. This one raises instead.

## Environment

`environment.yaml` pins pytorch 1.7 / cudatoolkit 11.0. `setup_env.sh` keeps
every other pin (**pytorch-lightning 1.4.2**, omegaconf 2.1.1, einops 0.3.0,
albumentations 0.4.3, …) and raises only the CUDA/torch pair to
**1.11.0+cu113** — exactly the bump CompVis themselves made in
`stable-diffusion`'s `environment.yaml`. cu113 is needed for sm_86; the
1.7/cu110 pin only reaches sm_80.

**GPU constraint.** `cuobjdump` on `torch/lib/libtorch_cuda_cu.so` reports SASS
for sm_35–sm_86 and, crucially, **no PTX at all** — so there is no JIT fallback
onto a newer device. It does not run slowly on sm_89+; it dies on the first
kernel launch with "no kernel image is available for execution on the device".

`gpu_c=8.0` is an *at-least* request (the complex relation is `<=`), so it
matches far more than it can run:

| type | cc | hosts | runs? |
|---|---|---|---|
| A100 | 8.0 | 9 | yes |
| A40 | 8.6 | 13 | yes |
| A6000 | 8.6 | 10 | yes |
| L40S / L40 | 8.9 | 30 | **no** |
| RTX6000ada | 8.9 | 4 | **no** |
| H200 | 9.0 | 10 | **no** |
| RTXP6000 | 12.0 | 8 | **no** |

32 of 84 matching hosts work. The run script requests plain `gpu_c=8.0` and
relies on `scripts/gpu_arch_guard.py` to abort in seconds on a bad landing. To
skip the bad landings entirely, add an allow-list — SGE honours alternation in a
RESTRING complex (verified: a nonexistent-type alternation drops every queue):

```bash
qsub -l 'gpu_type=(A40|A6000|A100)' scripts/run_ldm_ft_20k_scc.sh
```

```bash
qsub scripts/qsub_setup_env.sh      # builds ldm_official/venv_ldm (~10 min)
```
It must run on a compute node; SCC's login-node reaper kills the torch unpack.
`urllib3` is held below v2 because SCC's `python3/3.8.16` links OpenSSL 1.0.2
and urllib3 v2 refuses to import against anything below 1.1.1.

Two weight caches are pre-fetched so four DDP ranks do not race to download
them mid-run: torchvision's VGG16 backbone (`$TORCH_HOME`) and taming's LPIPS
head. The latter sits at `latent-diffusion/taming/modules/autoencoder/lpips/vgg.pth`
— a *cwd-relative* path baked into `taming.util.get_ckpt_path`, which is why it
lands inside the clone rather than in a cache dir. It is the only file in the
three clones that is not upstream's; md5 `d507d734…` matches taming's manifest.

## Data

`scripts/make_filelists.py` reproduces `distillation_nerc`'s split exactly —
`os.walk` → drop zero-byte → sort → `randperm(N, seed=42)` → 95/5 — so the LDM
baseline and the distillation runs are scored on the identical 1,000 held-out
images.

```bash
python scripts/make_filelists.py \
    --data_dir /projectnb/ivc-ml/mvasu6/LAION_FACE/subset_20k \
    --out_dir filelists/subset_20k
```

## Initial weights

LDM predates diffusers and uses the original CompVis key names (attention as
1×1 convs, up-blocks numbered in reverse). `scripts/convert_diffusers_vae_to_ldm.py`
inverts diffusers' `convert_ldm_vae_checkpoint` and also records a
deterministic reference forward pass, so `scripts/verify_converted_vae.py` can
prove from the LDM side that the two models are numerically the same network —
rather than trusting a hand-written key map.

```bash
# under venv_train (needs diffusers), already done:
../venv_train/bin/python scripts/convert_diffusers_vae_to_ldm.py \
    --model stable-diffusion-v1-5/stable-diffusion-v1-5 --subfolder vae \
    --out models/sd15_vae_ldm.ckpt
```

## Running

```bash
qsub scripts/smoke_test.sh                  # verify + 12 real steps on 1 GPU
qsub scripts/run_ldm_ft_20k_scc.sh       # 4×A40, 50 epochs
```

Run the smoke test first — it also pre-warms the LPIPS VGG download, which four
DDP ranks would otherwise race on. `BATCH=12 qsub scripts/smoke_test.sh` turns
it into a memory probe for the real run's per-GPU batch.

Measured on one A40 (48 GiB) at 256px, from those probes:

| per-GPU batch | peak memory | notes |
|---|---|---|
| 4 | 37.4 GiB | |
| 12 | 34.5 GiB | the configured batch — fits with room to spare |

Peak memory going *down* as the batch grows is not a typo. `benchmark: True`
(upstream's) leaves `cudnn.benchmark` on, and its algorithm search allocates
scratch workspaces that dominate the high-water mark and vary per shape, so the
peak is not activation-bound at these sizes. Lightning's `CUDACallback` resets
`max_memory_allocated` each epoch, so both figures are genuine within-epoch
peaks.

Lightning's parameter table is the check that the decoder-only split took:
**52.3 M trainable** (decoder 49.5 M + `post_quant_conv` + the 2.8 M
discriminator) against **48.9 M frozen** (encoder 34.2 M + LPIPS 14.7 M).

`pop from empty list` in the log is benign — upstream's own `print(e)` inside a
caught `IndexError` in `ImageLogger.check_frequency`.

`--scale_lr False` is deliberate. `main.py` otherwise sets
`lr = accum × ngpu × bs × base_lr`, turning 4.5e-6 into 8.6e-4. 4.5e-6 is upstream LDM's own base_lr; Stability publish no lr, so this is
LDM's value used unscaled rather than a figure taken from the model card. To
follow LDM's own scaling convention instead, drop the flag.

Overrides go on the command line — `main.py` merges unknown args via
`OmegaConf.from_dotlist`:

```bash
NUM_GPUS=2 EXTRA="lightning.trainer.accumulate_grad_batches=8" \
    qsub -l gpus=2 scripts/run_ldm_ft_20k_scc.sh   # keeps the 192 batch
```

### Deliberate departures that remain

Three settings would be actively harmful if reverted to upstream, so they are
left as-is rather than silently matched:

- **`--scale_lr False`.** Upstream defaults to `True`, i.e.
  `2 x 4 x 12 x 4.5e-6 = 4.32e-4`. Right for training kl-f8 from scratch;
  destructive for a decoder finetune, and Stability publish 4.5e-6.
- **Global batch is 96**, not the recipe's 192, now that
  `accumulate_grad_batches` is back at upstream's 2. Restore with
  `EXTRA="lightning.trainer.accumulate_grad_batches=4"`.
- **`max_epochs: 50`** — upstream sets none and would run forever.

### The discriminator

`disc_start: 50001` is upstream's and is left alone, but note what it means at
this dataset size. Each rank sees 4,750 images per epoch at batch 12 = 395
batches, and `accumulate_grad_batches: 4` makes that **~98 optimiser steps per
epoch** (confirmed on the smoke run: 12 batches at accum 1 reported
`global step 11`). Step 50001 therefore lands around **epoch 510** — ten times
the 50-epoch budget. **As configured, the GAN term never activates**, and the
run is a pure MSE + 0.1 · LPIPS decoder finetune. That happens to line up with the
`no_dis` distillation configs, which makes it the cleaner comparison.

To let the discriminator participate, lower it explicitly — e.g. warm up for
~10 epochs of a 50-epoch run:

```bash
EXTRA="model.params.lossconfig.params.disc_start=1000" \
    qsub scripts/run_ldm_ft_20k_scc.sh
```

## Output

Checkpoints, sample grids and tensorboard logs land in
`logs/<timestamp>_ldm_ftmse_20k/`. The saved weights are in **LDM** key layout;
converting back to diffusers for the existing eval pipeline means inverting the
map in `scripts/convert_diffusers_vae_to_ldm.py`.
