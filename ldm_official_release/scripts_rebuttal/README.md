# scripts_rebuttal

Everything added for the rebuttal, kept apart from `scripts/` so it is obvious what is new.

| file | what it does |
|---|---|
| `download_1M_data.sh` | fetch + unpack the 1M set into `$LDM_DATA` using `../split_0_1M_urls.txt` |
| `prepare_data_1M.sh` | teacher recons (skipped if downloaded), file lists, latent cache |
| `make_1M_signed_urls.sh` | regenerate the signed URLs once they expire (bucket owner only) |
| `prepare_sd35_teacher.sh` | fetch SD-3.5's VAE, write its 512 teacher recons, build the file lists |
| `run_dnarea.sh`, `run_gt256.sh`, `run_sd35.sh` | train one new arm per command; `SMOKE=1` for 12 steps |
| `../configs/pretune_1M_dnarea.yaml`, `../configs/pretune_1M_gt256.yaml`, `../configs/pretune_1M_sd35.yaml` | the three new arms |

## Run it end to end

```bash
export LDM_ROOT=$PWD LDM_DATA=/somewhere/with/1TB
bash setup_env.sh                                        # ./venv, ~10 min
bash scripts/prepare_weights.sh                          # ftmse_vae_ldm_with_disc.ckpt
bash scripts_rebuttal/download_1M_data.sh split_0_1M_urls.txt   # 591 GiB; ONLY=teacher skips the GT
bash scripts_rebuttal/prepare_data_1M.sh                 # lists + latent cache
bash scripts/train.sh configs/pretune_1M.yaml            # stage 1, 25 epochs
```

For the two new arms there is a runner each, so the last line becomes one of:

```bash
bash scripts_rebuttal/run_dnarea.sh      # SMOKE=1 for 12 steps first
bash scripts_rebuttal/run_gt256.sh
qsub -P <project> -l gpus=2 -l gpu_c=8.0 scripts_rebuttal/run_gt256.sh
```

The SD-3.5 arm needs its own teacher pass first, in place of `prepare_data_1M.sh`:

```bash
qsub -P <project> -l gpus=4 -l gpu_c=8.0 -pe omp 16 scripts_rebuttal/prepare_sd35_teacher.sh
bash scripts_rebuttal/run_sd35.sh
```

They are thin wrappers over `scripts/train.sh`; `NUM_GPUS` and `EXTRA` pass through. Stage 2 is `build_posttune_start_ckpt.py` then
`train.sh configs/posttune_1M.yaml`. [../DATA.md](../DATA.md) has the download in detail,
[../README.md](../README.md#the-1m-run) what makes the 1M path different from the 20k one.

Check it on a handful of images before committing a GPU-day:

```bash
BATCH=2 CONFIG=configs/pretune_1M_dnarea.yaml bash scripts/smoke_test.sh
```

## The two new arms

Both move the steerable comparison to 256, the student's own resolution, so **the
prediction is never resampled** — it is `pretune_1M`'s `match_mode: upsample` that lifts the
student 256 → 512.

- **`pretune_1M_dnarea`** keeps the 512 teacher reconstructions and brings them down by area
  averaging. `match_mode: downsample` had always used bicubic, a 4-tap kernel that overshoots
  at edges and so injects ringing into exactly the fine bands the steerable term measures.
  `down_mode: area` is new (default stays `bicubic`, so every existing config is unchanged —
  verified bit-identical). For the integer 2× step here torch's `area` **is** `cv2.INTER_AREA`:
  max |diff| 6e-8, float32 rounding. They are not interchangeable at non-integer ratios.
- **`pretune_1M_gt256`** drops the teacher entirely: the target is the ground-truth frame
  squashed straight to 256 (`teacher_from_source: True`, `teacher_size: 256`), so the loss
  resamples nothing at all.

### `pretune_1M_sd35` -- a different latent, same question

The nmn1M recipe with **SD-3.5 Large's VAE as both student and teacher**, so the model distils
from its own family exactly as `pretune_1M` has sd-vae-ft-mse distil from sd-vae-ft-mse. The
loss block is `pretune_1M`'s verbatim; the steerable term is untouched, because SD-3.5's VAE is
still f8 (`block_out_channels [128, 256, 512, 512]`, 4 downsampling stages) -- only the latent
is wider, 16 channels instead of 4. Student still decodes at 256 against a 512 teacher, so
height 5 still gives 22 bands and the graded ramp still means what it means elsewhere.

Four things change, and they are the whole diff:

- **Model class** `autoencoder_flux.AutoencoderKLFluxDecoderOnlyEMA`, which *holds* the
  diffusers module rather than converting it into ldm's `AutoencoderKL`. SD-3.5's VAE has no
  `quant_conv` and a 16-channel latent, so a conversion would mean inserting identity convs and
  re-keying every tensor. Despite the file name it is not FLUX-specific: `vae_repo` /
  `vae_class` pick the model.
- **`adam_betas: [0.9, 0.999]`, set explicitly.** This is load-bearing. `pretune_1M` leaves it
  unset and `AutoencoderKLSteerableEMANoGAN` then uses torch's defaults, but *this* class
  defaults to the GAN pair `(0.5, 0.9)`. Omitting it would make the comparison a betas confound
  instead of a latent one.
- **Its own teacher set and file lists.** `teacher_recons_512_png_sd35` and `split_0_1M_sd35`:
  ft-mse's recons would cap the steerable term at a weaker VAE's output, and a separate teacher
  pass drops a different handful of undecodable images.
- **No latent cache.** `split_0_1M_ftmse_*.npy` holds ft-mse's 4-channel latents; SD-3.5's
  encoder is a different function with a 16-channel output. The arm pays one frozen-encoder
  forward per step instead, which is why it may need a smaller micro-batch.

`scaling_factor 1.5305` / `shift_factor 0.0609` are not applied -- these losses live in image
space, as in every other arm here. Needs `diffusers` (`setup_env.sh` pins 0.30.3) and the VAE
from the ungated mirror of `stabilityai/stable-diffusion-3.5-large`, both handled by
`prepare_sd35_teacher.sh`.

### `pretune_1M_gt256` and `pretune_1M_dnarea`

Both set `steerable_height: 4` — at 256 the two finest 512 bands are above Nyquist, so
`('band',4,*)` does not exist — and **re-index the band weights one octave down** rather than
just dropping `band_4`. 256's `band_0` is the same octave as 512's `band_1`; deleting the top
band and leaving the rest would slide the whole ramp an octave coarse while looking unchanged.

None of the three has been trained. `pretune_1M_sd35` passes the suite's real training step
(builds the published VAE, z=16, encoder frozen, single optimiser, finite gradients that move).
The loss change is tested (defaults bit-identical, `area` ==
`INTER_AREA` at 2×, bad values rejected); the configs parse and resolve, but no run has used
them yet.
