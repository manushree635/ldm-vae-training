# LDM VAE decoder finetuning — portable copy

Finetune the Stable Diffusion VAE (kl-f8, 4-channel latent) with CompVis'
own [latent-diffusion](https://github.com/CompVis/latent-diffusion) trainer, from
plain `sd-vae-ft-*` style recipes up to teacher–student distillation with a
steerable-pyramid loss: the student decodes at 256, and is pulled towards what a
teacher VAE reconstructs at 512.

This is a self-contained copy of `ldm_official/`. Nothing in it points at a
particular machine or user: every path goes through two environment variables,
and one script builds each thing you need.

```
setup_env.sh            builds ./venv (one environment for everything)
scripts/env.sh          source this; sets LDM_ROOT, LDM_DATA, PYTHONPATH
scripts/prepare_weights.sh   initial checkpoints  -> models/
scripts/prepare_data.sh      crops, teacher recons, file lists -> $LDM_DATA, filelists/
scripts_rebuttal/prepare_data_1M.sh   the same for the 1M set (no crops, + latent cache)
scripts/smoke_test.sh        12 real steps on 1 GPU
scripts/train.sh             train any config (bash or qsub), also resumes
scripts/eval_recon.py        PSNR / LPIPS / rFID of a checkpoint
scripts/export_ldm_vae_to_diffusers.py   checkpoint -> diffusers AutoencoderKL dir

configs/pretune_1M.yaml, posttune_1M.yaml       the two-stage recipe
configs/pretune_1M_dnarea.yaml, _gt256.yaml     the two rebuttal arms
configs/snippets/       paste-in blocks: wandb logging, epoch-pinned checkpoints
ldm_ftmse/              all model / loss / data code written for this project
latent-diffusion/       upstream @ a506df5, main.py patched to run on Lightning 1.9
taming-transformers/    upstream @ 3ba01b2
```

## Quickstart

Needs Linux, an NVIDIA GPU, Python 3.9 or 3.10, ~10 GB of disk. The 20k configs
were run at per-GPU batch 12 on 48 GB A40s (peak ~35 GB; `benchmark: True` lets
cuDNN workspaces dominate, so smaller batches do not shrink the peak much —
probe yours with `BATCH=<n> bash scripts/smoke_test.sh`). The `*_h200`/1M configs
assume 141 GB cards.

```bash
cd ldm_official_release

# 1. environment (~10 min). On BU SCC run it on a compute node -- see the script header.
bash setup_env.sh
source scripts/env.sh

# 2. initial weights: SD-1.5 VAE, sd-vae-ft-mse, and CompVis' pretrained discriminator
bash scripts/prepare_weights.sh

# 3. data: point it at ANY folder of images (needs 1 GPU for the teacher passes)
SRC_DIR=/path/to/images bash scripts/prepare_data.sh

# 4. check the whole path works before spending a GPU-day
CONFIG=configs/pretune_1M_dnarea.yaml bash scripts/smoke_test.sh

# 5. train
CONFIG=configs/pretune_1M_dnarea.yaml NUM_GPUS=2 bash scripts/train.sh

# 6. score it
python scripts/eval_recon.py --ckpt logs/<run>/checkpoints/last.ckpt --use_ema \
    --filelist filelists/subset_20k_rcrop/val.txt
```

Steps 1–3 are one-time. Configs 01 and 02 need only the split from step 3, not
the crops or the teacher passes.

### On BU SCC

The launchers carry SGE headers, so `qsub` replaces `bash`. Two rules:

- **Submit from the repo root.** The jobs use `-cwd` to find the repo (SGE copies
  the script to a spool dir, so the script cannot locate itself).
- **Pass variables with `-v`.** `CONFIG=x qsub scripts/train.sh` does *not* forward
  `CONFIG`; the job silently trains the default config instead.

```bash
qsub -P <project> -l h_rt=2:00:00 -pe omp 4 -cwd setup_env.sh
qsub -P <project> -l gpus=1 -l gpu_c=7.0 -v SRC_DIR=/path/to/images scripts/prepare_data.sh
qsub -P <project> -N w1graded -l gpus=2 -l gpu_c=8.0 \
     -v CONFIG=configs/pretune_1M_dnarea.yaml,NUM_GPUS=2 scripts/train.sh
```

Keep `-l gpus=` equal to `NUM_GPUS`. The default venv (torch 2.4.1+cu121) runs on
every SCC GPU from V100 to H200; for RTX PRO 6000 (sm_120) build with
`TORCH_VARIANT=cu128 bash setup_env.sh`. `scripts/gpu_arch_guard.py` aborts a job
in seconds if it lands on a GPU the build has no kernels for.

If you can read the original author's project space, you can skip steps 2–3 and
reuse the existing data: `export LDM_DATA=/projectnb/ivc-ml/mvasu6/LAION_FACE`
before `source scripts/env.sh`. The `models/*.ckpt` symlinks and `filelists/`
shipped in this directory already point there; on any other machine delete them
and run steps 2–3.

## How paths work

| variable | meaning | default |
|---|---|---|
| `LDM_ROOT` | this directory | derived by `scripts/env.sh` from its own location |
| `LDM_DATA` | image sets and teacher reconstructions | `$LDM_ROOT/data` — export it *before* sourcing `env.sh` to override |
| `LDM_VENV` | virtualenv to activate | `$LDM_ROOT/venv` |
| `HF_HOME`, `TORCH_HOME` | download caches | `$LDM_ROOT/.cache/...` unless already set |

Configs read them through OmegaConf: `ckpt_path: ${oc.env:LDM_ROOT}/models/...`,
`teacher_dir: ${oc.env:LDM_DATA}/...`. `scripts/check_config_paths.py <config>`
resolves a config and tells you which input is missing and which script builds
it; `train.sh` and `smoke_test.sh` run it first.

The file lists under `filelists/` hold **absolute** image paths, so they are
machine-specific — regenerate them (step 3) rather than copying them around.
That includes the `filelists/split_0_1M/` lists shipped here: they point at the
machine this copy was made on, so the 1M configs cannot run until you rebuild
them.

## The 1M run

`configs/pretune_1M.yaml` is the full-scale recipe — L1 1.0 + VGG-LPIPS 1.0 +
steerable 1.0, no GAN, all 22 graded bands from epoch 0, 25 epochs — and
`configs/posttune_1M.yaml` is the 5-epoch adversarial stage that follows it.
`scripts/prepare_data.sh` does **not** build what they read; use:

If you do not already have the images, [DATA.md](DATA.md) has the download: the set is
staged in GCS as tar shards behind 7-day signed URLs, and unpacking it lets you skip the
teacher pass entirely.

```bash
SRC_DIR=/path/to/your/1M/images bash scripts_rebuttal/prepare_data_1M.sh
bash scripts/train.sh configs/pretune_1M.yaml
python scripts/build_posttune_start_ckpt.py --help   # stage 1 -> stage 2 start
bash scripts/train.sh configs/posttune_1M.yaml
```

Three ways the 1M path differs from the 20k one, each of which will break a run
if you assume otherwise:

- **No crops.** The 20k configs train on one 512×512 random crop per image. The
  1M configs train on the *whole frame* squashed to 256×256 against a teacher
  recon of the whole frame squashed to 512×512 (`ldm_ftmse/data_squash.py`).
  Feeding them the rcrop directories misaligns every non-square image.
- **The file lists are an intersection.** `make_filelists_teacher.py` keeps only
  images that have a teacher recon, because `verify_teacher` aborts the run on a
  missing one and at this scale a few images always fail to decode.
- **There is a latent cache.** The encoder is frozen and the crop deterministic,
  so its output is cached once rather than recomputed each epoch — bit-exact, but
  keyed to the file lists. Change the lists and you must rebuild it; a stale cache
  would otherwise train on the wrong latents. `precompute_latents.py` writes a
  fingerprint sidecar and the dataset refuses a mismatch.

Budget for it: the teacher pass is about **1 GPU-day** and **350 GB** of PNGs for
1M images on an A40, the latent cache about **32 GB**, and training is ~2.1 h per
epoch on 2×H200. Every prep step skips work already done, so an interrupted run
can be resubmitted as-is.

## Choosing and changing a config

This branch carries only the configs the rebuttal needs: `pretune_1M.yaml` and
`posttune_1M.yaml` are the two-stage recipe, and `pretune_1M_dnarea.yaml` /
`pretune_1M_gt256.yaml` are the two new arms, described in
[scripts_rebuttal/README.md](scripts_rebuttal/README.md). The 12 starter configs and
the 139 archived research configs were removed here; they are on the
`portable-release` branch if you need them.

To change a setting without editing a file, pass OmegaConf
overrides through `EXTRA` (space-separated):

```bash
CONFIG=configs/pretune_1M_dnarea.yaml NUM_GPUS=2 \
EXTRA="data.params.batch_size=8 lightning.trainer.max_epochs=10 model.params.lossconfig.params.steerable_weight=2.0" \
    bash scripts/train.sh
```

To make a new recipe, copy the closest starter config and edit it. The knobs
that matter are in `model.params.lossconfig.params`:

| key | what it does |
|---|---|
| `rec_loss_type` (`l1` / `mse`), `perceptual_weight` | pixel term and LPIPS weight |
| `disc_start`, `disc_weight` | step at which the GAN term turns on, and its weight |
| `steerable_weight` | weight of the teacher steerable-pyramid term (0 = off) |
| `steerable_band_weights` | per-band weights; level 0 is the FINEST scale |
| `steerable_complex_metric` | `magnitude` (phase-blind) or a phase-aware variant |
| `steerable_band_schedule`, `steerable_band_schedule_order` | activate bands over epochs (`fine_to_coarse` / `coarse_to_fine`) instead of all at once |

What the steerable term distils *towards* is set on the data side: `teacher_dir` +
`teacher_ext` name the 512px target of each image — the teacher's reconstructions
(default) or the ground-truth crops themselves (config 08).
`model.target` picks the model class (decoder-only, EMA + GAN, no-GAN, full
encoder). `model.base_learning_rate` is the learning rate actually used —
`train.sh` passes `--scale_lr False`, because upstream's automatic scaling
(`accum × gpus × batch × lr`) is right for training from scratch and destructive
for a finetune.

**Global batch** = `batch_size × NUM_GPUS × accumulate_grad_batches`. Most 20k
configs were run at 12 × 4 × 2 = 96 (config 09: 12 × 2 × 16 = 384). If you change the GPU count, change
`accumulate_grad_batches` to keep the product, or accept a different recipe.

## Resuming, checkpoints, logging

```bash
RESUME_DIR=logs/<run> NUM_GPUS=2 bash scripts/train.sh         # continues epoch numbering
```

- `logs/<timestamp>_<RUN_NAME>/checkpoints/` holds `last.ckpt` plus the top 3 by
  `val/rec_loss`. **That monitor is anti-correlated with rFID on GAN runs**, so the
  top-3 rule tends to delete the late epochs you want. Paste
  `configs/snippets/periodic_ckpt_snippet.yaml` into a config to pin epochs
  (~1 GB each), and choose checkpoints with `scripts/eval_recon.py`.
- Models whose class name ends in `EMA` keep EMA weights inside the checkpoint;
  score and export them with `--use_ema` / `--use-ema`.
- Tensorboard is the default logger (`tensorboard --logdir logs`). All four 1M
  configs log to Weights & Biases: run `wandb login` first,
  or `export WANDB_MODE=offline`. `configs/snippets/wandb_snippet.yaml` adds it to
  any config.
- Export for anything that speaks diffusers:
  `python scripts/export_ldm_vae_to_diffusers.py --ckpt logs/<run>/checkpoints/last.ckpt --use-ema --out models/<name>_diffusers`

## Things that will bite you

- **Teacher alignment.** The teacher configs require square 512×512 sources, so
  the student's 256 input is exactly the area-downsampled teacher input. That is
  why `prepare_data.sh` makes crops, and why the dataset class refuses random
  cropping. Details in `ldm_ftmse/data_teacher.py`.
- **Do not re-split the crop set.** `make_rcrop_filelists.py` maps the base split
  onto the crops by filename. Re-running `make_filelists.py` on the crop folder
  moves ~5% of images between train and val (the `_rcrop` suffix changes the sort
  order the seed is applied to) — silent leakage against every other run.
- **`precision: 16` + Trainer `gradient_clip_val`** clips *scaled* gradients under
  this Lightning version. The no-GAN model class (`autoencoder_nogan.py`) has its own
  `grad_clip_norm` parameter, which clips the true gradients; use that instead.
- **`latents_file` (configs 11 and the `*_h200` ones)** caches the frozen encoder's
  output. Build it with `scripts/precompute_latents.py`; it must stay **off** for
  any config that trains the encoder (config 10).
- **Multi-GPU hangs on PCIe-only nodes.** `scripts/nccl_env.sh` disables P2P/SHM,
  and `scripts/ddp_preflight.py` proves an all-reduce completes before the run starts.
- `pop from empty list` in the log is benign — upstream's own `print(e)`.
- `assets/fixed_recon_image.jpg` is only the pinned picture the wandb panel
  reconstructs over time; swap in any image.

## What is not here

The research tree's evaluation harnesses (COCO rFID via `vae_bench`, TokBench,
FAN landmarks, PIE-Bench, …) live in other directories with their own
environments and were not copied; `scripts/eval_recon.py` is the self-contained
replacement. Results quoted in config comments come from those harnesses and are
not reproducible from this directory alone. Run-specific launchers, analysis
scripts, logs and trained weights were left behind too. `CLIP/` is not needed for
VAE training.

`README_research.md` is the original design document: how the ft-MSE recipe was
mapped onto LDM, how the trainable set was measured, and why each deliberate
departure from upstream exists.
