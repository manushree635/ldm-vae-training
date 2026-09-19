# Configs

Every config here is the stock kl-f8 architecture (`ddconfig` is never touched);
they differ in the initial weights, the loss, which parameters train, and the data
pipeline. All paths go through `${oc.env:LDM_ROOT}` / `${oc.env:LDM_DATA}`.

## Starter set

Each is a verbatim copy of one research config (right-hand column, under
`experiments/`, minus the `autoencoder_kl_32x32x4_` prefix) — read its header
comment for the full reasoning.

| config | init | recipe | what it is for | source |
|---|---|---|---|---|
| `01_ftema_baseline` | SD-1.5 VAE | L1 + 1.0·LPIPS, decoder-only, upstream loss verbatim | the plainest run; the sd-vae-ft-ema recipe | `ftema_20k` |
| `02_ftmse_baseline` | SD-1.5 VAE | MSE + 0.1·LPIPS, decoder-only | the sd-vae-ft-mse recipe | `ftmse_20k` |
| `03_gan_control` | ft-mse | L1 + LPIPS + EMA + pretrained GAN, **no** steerable term | the control every steerable arm is compared against (same data, same GAN) | `emagan_ftmse_ctrl_20k` |
| `04_steerable_graded` | ft-mse | 03 + steerable 1.0, graded band weights (0.12 fine → 1.50 coarse) | the main distillation recipe ("w1graded") | `emagan_ftmse_w1graded_20k` |
| `05_steerable_equal` | ft-mse | 03 + steerable 1.0, all bands weighted equally | isolates the band weighting | `emagan_ftmse_w1eq_20k` |
| `06_steerable_fine_to_coarse` | ft-mse | 04 with bands switched on one scale at a time | isolates the band schedule | `emagan_ftmse_w1f2c_20k` |
| `07_steerable_nogan` | ft-mse | 04 with the adversarial term removed (one optimizer) | isolates the GAN; ~1.5× faster per epoch | `ftmse_w1graded_nogan_20k` |
| `08_steerable_gt_target` | ft-mse | 04, but the steerable target is the ground-truth 512 crop, not the teacher's recon | teacher vs ground truth as the distillation target | `emagan_ftmse_w1graded_gt_20k` |
| `09_nercmatch_gan_l1vgg1_str05` | ft-mse | L1 + 1.0·VGG-LPIPS + 0.5·steerable, mean-reduced loss, no KL, fp16, global batch 384, GAN from epoch 2, squash-resize data | a different loss *scaling* and data geometry; lowest COCO rFID of the 20k arms in the original study | `nercmatch_gan_l1vgg1_str05_20k` |
| `10_fullenc_magstr_bandbal` | ft-mse | encoder **and** decoder train; magnitude steerable + band-balanced MSE 0.25; latent anchored to the original (std 0.1); bf16 | the encoder-unfrozen line of work | `emagan_ftmse_fullenc_magstr_bbteach025_20k_h200` |
| `11_steerable_graded_1M` | ft-mse | 04 scaled to a ~1M-image set, batch 48/GPU, bf16, cached encoder latents | template for a large run; needs your own 1M lists, teacher recons and latent cache | `emagan_ftmse_w1graded_1M_h200` |
| `12_dual_branch_1M` | ft-mse | two-tail decoder, `out = LP(b1) + HP(b2)`: branch 1 on L1 + 1.0·VGG-LPIPS, branch 2 on steerable 0.25 (levels 1–2 only), no GAN, mean-reduced loss; 1M data block of 11 | the best 20k dual-branch variant ("replace, noL0hp") scaled to 1M. **Not yet run**: every dual arm lost to its single-head control at 20k (rFID 16.7 vs 11.6), this asks whether data fixes it. Batch 48 needs an H200 | `nercmatch_l1vgg1_str025_noL0hp_dual_replace_20k` (model) + `nercmatch_l1vgg1_str1_1M_h200` (data/trainer) |

What each needs on disk (`scripts/check_config_paths.py <config>` tells you exactly):

| configs | checkpoint | data |
|---|---|---|
| 01, 02 | `models/sd15_vae_ldm.ckpt` | `filelists/subset_20k` |
| 03–08, 10 | `models/ftmse_vae_ldm_with_disc.ckpt` | `filelists/subset_20k_rcrop` + `$LDM_DATA/subset_20k_rcrop_teacher_recons_512_png` (08: the crops themselves) |
| 09 | same | `filelists/nerc_subset_20k` + `$LDM_DATA/subset_20k_teacher_recons_512_png`; logs to wandb |
| 11, 12 | same | `filelists/split_0_1M`, `$LDM_DATA/teacher_recons_512_png`, `$LDM_DATA/latent_cache/*.npy` — **not** built by `prepare_data.sh` |

`prepare_weights.sh` and `prepare_data.sh` build everything in the first three rows.

### Three data pipelines, and why they are not interchangeable

| dataset class | student input | teacher target | used by |
|---|---|---|---|
| `ldm_ftmse.data.FileListSR*` | upstream's random-resized crop → 256 | none | 01, 02 |
| `ldm_ftmse.data_teacher.FileListSRTeacher*` | full frame of a **square 512** source, area-downsampled to 256; cropping is forced off | `<teacher_dir>/<stem><teacher_ext>` at 512 | 03–08, 10 |
| `ldm_ftmse.data_nerc.*` / `data_squash.*` | anisotropic squash of the full frame to 256 | recon of the same squash at 512 | 09 / 11, 12 |

A teacher recon only means something if the student sees the same pixels it was
computed from. Pair the dataset class with teacher recons built for its geometry.

## If you are trying to reproduce a number from the original study

- **Lightning version changes `disc_start`.** Configs 01–08 were written for
  Lightning 1.4.2, where `global_step` counts optimizer steps once. Under the
  1.9.5 this copy installs, `global_step` advances for *both* optimizers of a GAN
  run, so the discriminator switches on after half as many batches. Double
  `disc_start` to keep the original timing. Configs 09–11 (and anything named
  `*_h200`) were already written for 1.9.5.
- **The stack itself moves rFID.** The same config re-run on the newer
  torch/Lightning stack scored about 0.5 COCO-rFID worse than on the original
  torch 1.11 / Lightning 1.4.2 stack. Compare runs within one stack.
- Results quoted in config comments are COCO rFID (n=2000) from an evaluation
  harness that is not part of this copy. `scripts/eval_recon.py` on your val list
  ranks runs against each other; its absolute values are not comparable to those.
- Learning rates are per-config (`4.5e-6` or `1e-5`); a few header comments
  predate a change to the value below them. The YAML value is what runs.

## `experiments/`

All 139 configs from the research runs, unedited except for the path rewrite.
Naming, after the `autoencoder_kl_32x32x4_` prefix:

| fragment | meaning |
|---|---|
| `emagan` | EMA weights + GAN on from ~epoch 1 with the pretrained discriminator |
| `ftmse` / `ftema` | initialised from sd-vae-ft-mse / ft-ema (absent = SD-1.5 VAE) |
| `w1`, `w2`, `w5`, `w10`, `w05` | steerable weight 1.0, 2.0, 5.0, 10.0, 0.5 |
| `graded` / `eq` / `rev` / `mid` | band-weight profile: coarse-heavy ramp / equal / reversed / mid-band |
| `f2c` | fine-to-coarse band schedule |
| `dn` | student and teacher compared at 256 (`steerable_match_mode: downsample`) instead of upsampling the student to 512 |
| `post` | post-tune: continue a steerable run with the steerable term switched off |
| `cplx`, `hyb` | phase-aware / hybrid complex steerable metric instead of magnitude |
| `subpix`, `pool3` | sub-pixel upsampler head in the loss / 3×3 pooled band magnitudes |
| `gt`, `gt256` | steerable target is ground truth rather than the teacher |
| `nogan`, `frozendisc` | no adversarial term / discriminator not updated |
| `nercmatch`, `nercrecipe` | mean-reduced loss scaling + squash data, matching the sibling `distillation_nerc` codebase |
| `dual_*` | two-branch decoder (`autoencoder_dualbranch.py`); score with `eval_recon.py --config <the yaml>`, export with `export_dual_branch_to_diffusers.py` |
| `fullenc`, `stage2dec`, `bb*` | encoder unfrozen / decoder stage after it / band-balanced MSE |
| `dino` | adds a DINOv2 feature loss |
| `10img_*`, `bottleneck_probe`, `ab_*`, `bench_*` | overfit probes, diagnostics, speed A/Bs — not recipes |
| `20k` / `1M`, `h200` / `p6k` | dataset size; the GPU the batch size and precision were tuned for |

Some of these point at inputs this copy cannot build (1M lists, latent caches,
checkpoints from other runs such as `strpy1M_*` or `stage1_G11_*`);
`check_config_paths.py` will say so before anything runs.

## `snippets/`

`wandb_snippet.yaml` and `periodic_ckpt_snippet.yaml` are `lightning:` blocks to
paste into any config.
