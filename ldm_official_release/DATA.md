# Getting the 1M training data

The 1M face set this work trains on — source frames, their teacher reconstructions,
and the train/val split — is staged as tar shards in Google Cloud Storage:

```
gs://vae-distill/split_0_1M_tar/          591 GiB, 201 objects
  meta.tar                                 14 MB   README + split_0_1M/{train,val}.txt
  gt_000_of_100.tar  … gt_099_of_100.tar   247 GB  source frames  -> images/<stem>.jpg
  teacher_000_of_100.tar … teacher_099_of_100.tar
                                           352 GB  teacher recons -> teacher/<stem>.png
```

The two sets **pair by stem**: `teacher/<stem>.png` is the sd-vae-ft-mse reconstruction of
`images/<stem>.jpg`. There is no separate mapping file and none is needed.

## Getting a download link

The bucket is private; access is handed out as **signed URLs**. The list is in this repo:

```
split_0_1M_urls.txt        201 links -- 100 gt + 100 teacher + meta
```

**These links expire 2026-10-31** (7 days is the maximum `gcloud` will sign for) and each one
downloads without any sign-in, so treat the file as a credential and do not forward it. Once
it has expired, regenerate with bucket access:

```bash
GCS_SIGNER_SA=<signer-sa> bash scripts_rebuttal/make_1M_signed_urls.sh split_0_1M_urls.txt
```

A signed URL cannot be revoked individually -- to cut access early the objects have to be
deleted or the signing key rotated.

## Download and unpack

```bash
mkdir -p $LDM_DATA && cd $LDM_DATA
# -P 4 keeps four transfers in flight; drop it if your link is the bottleneck
xargs -a $LDM_ROOT/split_0_1M_urls.txt -P 4 -I{} sh -c 'curl -fsSL "{}" | tar -x'
```

That lands `images/`, `teacher/`, `split_0_1M/` and `README.txt` under `$LDM_DATA`. Check
you got everything before going further — a truncated tar fails quietly in a parallel loop:

```bash
ls images  | wc -l     # expect 992574
ls teacher | wc -l     # expect 992574
```

## Point the training at it

The 1M configs expect `$LDM_DATA/split_0` and `$LDM_DATA/teacher_recons_512_png`, so either
rename on unpack or symlink:

```bash
ln -s "$LDM_DATA/images"  "$LDM_DATA/split_0"
ln -s "$LDM_DATA/teacher" "$LDM_DATA/teacher_recons_512_png"
```

The shipped `split_0_1M/{train,val}.txt` hold **basenames only** — our own copies held
absolute paths, which are useless on another machine. The training config reads absolute
paths, so rebuild the lists and the latent cache on your side rather than editing them:

```bash
bash setup_env.sh                                  # builds ./venv
bash scripts/prepare_weights.sh                    # ftmse_vae_ldm_with_disc.ckpt
bash scripts_rebuttal/prepare_data_1M.sh                    # lists + latent cache; skips the
                                                   # teacher pass, already downloaded
bash scripts/train.sh configs/pretune_1M.yaml      # stage 1, 25 epochs
python scripts/build_posttune_start_ckpt.py --help # stage 1 -> stage 2 start
bash scripts/train.sh configs/posttune_1M.yaml     # stage 2, adversarial
```

`prepare_data_1M.sh` re-runs the teacher pass only if `teacher_recons_512_png` is missing,
so with the download in place it goes straight to the file lists and the latent cache
(~32 GB, a couple of hours). See the **The 1M run** section of [README.md](README.md) for
what makes the 1M path different from the 20k one — no crops, intersected file lists, and a
latent cache keyed to those lists.

## Disk

About **1 TB** all told: 599 GB of tars, the same again unpacked unless you delete each tar
as it extracts, plus 32 GB of latent cache. The `curl | tar -x` pipeline above never writes
the tar, so it needs only the unpacked size.

## Provenance, and one caveat worth reading

These are LAION-Face images, web-scraped, redistributed here only for reproducing this work.

**14.43% of the 992,574 files are absent from Re-LAION**, the safety-revised re-release of
LAION-5B — roughly 143k images the dataset maintainers withdrew. We checked that this is
neither link rot (87.6% of the URLs still resolve) nor a string-matching artifact. If you
need a set that tracks Re-LAION, intersect against the Re-LAION ids before using it.

The teacher reconstructions are ours: each source frame squashed to 512×512 (one anisotropic
resize, **no crop**) and round-tripped through `stabilityai/sd-vae-ft-mse`, taking the
posterior mean. The student trains on the same frame squashed to 256. Pairing these teachers
with a centre-crop pipeline misaligns every non-square image.
