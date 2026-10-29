# scripts_rebuttal

Everything added for the rebuttal, kept apart from `scripts/` so it is obvious what is new.

| file | what it does |
|---|---|
| `download_1M_data.sh` | fetch + unpack the 1M set into `$LDM_DATA` using `../split_0_1M_urls.txt` |
| `prepare_data_1M.sh` | teacher recons (skipped if downloaded), file lists, latent cache |
| `make_1M_signed_urls.sh` | regenerate the signed URLs once they expire (bucket owner only) |
| `prepare_sd35_teacher.sh` | fetch SD-3.5's VAE, write its 512 teacher recons, build the file lists |
| `prepare_flux2_teacher.sh` | the same for FLUX.2 (needs the overlay below, and HF login) |
| `install_diffusers036.sh` | one-off: diffusers 0.36 beside the venv, for `AutoencoderKLFlux2` |
| `run_dnarea.sh`, `run_gt256.sh`, `run_sd35.sh`, `run_flux2.sh` | train one new arm per command; `SMOKE=1` for 12 steps |
| `../configs/pretune_1M_{dnarea,gt256,sd35,flux2}.yaml` | the four new arms |

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

The SD-3.5 and FLUX.2 arms each need their own teacher pass, in place of `prepare_data_1M.sh`:

```bash
qsub -P <project> -l gpus=4 -l gpu_c=8.0 -pe omp 16 scripts_rebuttal/prepare_sd35_teacher.sh
bash scripts_rebuttal/run_sd35.sh

bash scripts_rebuttal/install_diffusers036.sh          # once, for FLUX.2 only
huggingface-cli login                                  # FLUX.2-dev is gated
qsub -P <project> -l gpus=4 -l gpu_c=8.0 -pe omp 16 scripts_rebuttal/prepare_flux2_teacher.sh
bash scripts_rebuttal/run_flux2.sh
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

### `pretune_1M_sd35` and `pretune_1M_flux2` -- a latent-width ladder

The nmn1M recipe with **SD-3.5 Large's** and **FLUX.2's** VAE as both student and teacher, so each
model distils from its own family exactly as `pretune_1M` has sd-vae-ft-mse distil from
sd-vae-ft-mse. The loss block is `pretune_1M`'s verbatim in both, and the steerable term is
untouched, because all three VAEs are **f8** with the same `block_out_channels
[128, 256, 512, 512]` and 4 downsampling stages. Only the latent width changes:

| arm | latent | `vae_class` | quant_conv | at 256 input |
|---|---|---|---|---|
| `pretune_1M` | 4 ch | (ldm-native) | yes | 32x32x4 |
| `pretune_1M_sd35` | 16 ch | `AutoencoderKL` | **no** | 32x32x16 |
| `pretune_1M_flux2` | 32 ch | `AutoencoderKLFlux2` | yes | 32x32x32 |

So the pair isolates latent **width** at a fixed spatial rate. Each student still decodes at 256
against a 512 teacher, so height 5 still gives 22 bands and the graded ramp still means what it
means elsewhere. Measured geometry, not assumed: FLUX.2's encoder is 34.4M and its decoder
49.6M, and it carries no `scaling_factor`/`shift_factor` at all, where SD-3.5 has both (1.5305 /
0.0609, unapplied -- these losses live in image space).

Four things change, and they are the whole diff:

- **Model class** `autoencoder_flux.AutoencoderKLFluxDecoderOnlyEMA` for both, which *holds* the
  diffusers module rather than converting it into ldm's `AutoencoderKL`. SD-3.5's VAE has no
  `quant_conv` and a 16-channel latent, so a conversion would mean inserting identity convs and
  re-keying every tensor. Despite the file name it is not FLUX-specific: `vae_repo` /
  `vae_class` pick the model, and it handles the presence or absence of `quant_conv`
  generically -- it freezes encoder + `quant_conv` and optimises decoder + `post_quant_conv`.
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

**Stacks differ, and FLUX.2's is not optional.** SD-3.5 runs on the pinned `diffusers==0.30.3`
and takes its VAE from the ungated mirror of `stabilityai/stable-diffusion-3.5-large`, both
handled by `prepare_sd35_teacher.sh`. `AutoencoderKLFlux2` only exists from **diffusers 0.36**,
so rather than move the library under every other arm, `install_diffusers036.sh` installs 0.36
`--no-deps` into `pylibs/diffusers036` and `EXTRA_PYTHONPATH` (new in `scripts/env.sh`) puts it
ahead of the venv for that run only. 0.36 is the last release supporting python 3.9 and needs
torch >= 2.4; both `setup_env.sh` variants satisfy that, but a torch 1.x env cannot run this arm
at all -- the import dies on `torch.xpu`. `black-forest-labs/FLUX.2-dev` is also **gated**, so
`huggingface-cli login` first.

### `pretune_1M_gt256` and `pretune_1M_dnarea`

Both set `steerable_height: 4` — at 256 the two finest 512 bands are above Nyquist, so
`('band',4,*)` does not exist — and **re-index the band weights one octave down** rather than
just dropping `band_4`. 256's `band_0` is the same octave as 512's `band_1`; deleting the top
band and leaving the rest would slide the whole ramp an octave coarse while looking unchanged.

None of the four has been trained. Both new VAE arms pass the suite's real training step --
`pretune_1M_sd35` builds the published VAE at z=16 with 49.5M trainable decoder params,
`pretune_1M_flux2` at z=32 with 49.6M, each freezing its 34.3-34.4M encoder, each on a single
optimiser with finite gradients that move.
The loss change is tested (defaults bit-identical, `area` ==
`INTER_AREA` at 2×, bad values rejected); the configs parse and resolve, but no run has used
them yet.
