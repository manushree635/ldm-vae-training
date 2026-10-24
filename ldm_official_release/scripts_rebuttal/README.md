# scripts_rebuttal

Everything added for the rebuttal, kept apart from `scripts/` so it is obvious what is new.
Nothing in `scripts/` was changed; these four files are additive.

| file | what it does |
|---|---|
| `download_1M_data.sh` | fetch + unpack the 1M set from signed URLs into `$LDM_DATA` |
| `prepare_data_1M.sh` | teacher recons (skipped if downloaded), file lists, latent cache |
| `make_1M_signed_urls.sh` | regenerate 7-day signed URLs (bucket owner only) |
| `../configs/pretune_1M_dnarea.yaml`, `../configs/pretune_1M_gt256.yaml` | the two new arms |

## Run it end to end

```bash
export LDM_ROOT=$PWD LDM_DATA=/somewhere/with/1TB
bash setup_env.sh                                        # ./venv, ~10 min
bash scripts/prepare_weights.sh                          # ftmse_vae_ldm_with_disc.ckpt
bash scripts_rebuttal/download_1M_data.sh urls.txt       # 591 GiB; ONLY=teacher to skip the GT
bash scripts_rebuttal/prepare_data_1M.sh                 # lists + latent cache
bash scripts/train.sh configs/pretune_1M.yaml            # stage 1, 25 epochs
```

`scripts/train.sh` takes any config, so swap the last line for `pretune_1M_dnarea.yaml` or
`pretune_1M_gt256.yaml`. Stage 2 is `build_posttune_start_ckpt.py` then
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

Both set `steerable_height: 4` — at 256 the two finest 512 bands are above Nyquist, so
`('band',4,*)` does not exist — and **re-index the band weights one octave down** rather than
just dropping `band_4`. 256's `band_0` is the same octave as 512's `band_1`; deleting the top
band and leaving the rest would slide the whole ramp an octave coarse while looking unchanged.

Neither has been trained. The loss change is tested (defaults bit-identical, `area` ==
`INTER_AREA` at 2×, bad values rejected); the configs parse and resolve, but no run has used
them yet.
