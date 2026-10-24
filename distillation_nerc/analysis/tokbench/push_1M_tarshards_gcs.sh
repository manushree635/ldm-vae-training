#!/bin/bash -l
# Tar the 1M training set into shards and stream them straight to GCS (no local copy, so
# this needs no extra disk). Shards make the set downloadable WITHOUT per-object access:
# a handful of tars can be handed out as signed URLs.
#
#   qsub -t 1-100 -v SET=gt      push_1M_tarshards_gcs.sh     # 247 GB of source JPEGs
#   qsub -t 1-100 -v SET=teacher push_1M_tarshards_gcs.sh     # 352 GB of teacher PNGs
#   bash push_1M_tarshards_gcs.sh meta                        # file lists + manifest (fast, no qsub)
#
# Resumable: a task whose shard already exists in the bucket exits immediately, so a
# killed array can simply be resubmitted.
#
# THE TWO SETS PAIR BY STEM. teacher_recons_512_png/<stem>.png is the sd-vae-ft-mse
# reconstruction of split_0/<stem>.jpg, squashed to 512x512. There is no separate mapping
# file and none is needed -- the stem IS the key. The meta shard carries the train/val
# lists, which are what actually define the run.
#$ -P ivc-ml
#$ -pe omp 2
#$ -l h_rt=12:00:00
#$ -N push_1M
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/distillation_nerc/logs
set -euo pipefail
LF=/projectnb/ivc-ml/mvasu6/LAION_FACE
FL=/projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/filelists/split_0_1M
BUCKET=gs://vae-distill
SET="${SET:-${1:-}}"
: "${SET:?set SET=gt | teacher | meta}"
module load python3/3.10.12 google-cloud-sdk/455.0.0

if [ "$SET" = "meta" ]; then
    OUT="$BUCKET/split_0_1M_tar/meta.tar"
    if gcloud storage ls "$OUT" >/dev/null 2>&1; then echo "exists, skipping: $OUT"; exit 0; fi
    STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
    mkdir -p "$STAGE/split_0_1M"
    # rebase the lists: ours hold absolute paths on THIS machine, which are useless to
    # anyone else. Ship basenames; the README says what to prepend.
    for f in train val; do
        sed 's#.*/##' "$FL/$f.txt" > "$STAGE/split_0_1M/$f.txt"
    done
    cat > "$STAGE/README.txt" <<'TXT'
1M face set used for the two-stage VAE decoder distillation.

  images/<stem>.jpg   source image                        (shards gt_NNN_of_NNN.tar)
  teacher/<stem>.png  sd-vae-ft-mse reconstruction, 512^2 (shards teacher_NNN_of_NNN.tar)
  split_0_1M/train.txt, val.txt   990,574 / 2,000 basenames

The two sets pair BY STEM: teacher/<stem>.png is the reconstruction of images/<stem>.jpg.
The file lists hold basenames only -- prepend your own image directory. The training
config reads absolute paths, so regenerate the lists on your side (the release copy's
scripts/prepare_data_1M.sh does this) rather than editing these by hand.

The teacher reconstructions were produced by squashing each full frame to 512x512 (one
anisotropic resize, NO crop) and round-tripping it through stabilityai/sd-vae-ft-mse.
The student trains on the same frame squashed to 256x256. Pairing these teachers with a
centre-crop pipeline misaligns every non-square image.

PROVENANCE AND A CAVEAT WORTH READING
These are LAION-Face images, web-scraped and redistributed here only for reproducing this
work. 14.43% of these 992,574 files are absent from Re-LAION, the safety-revised re-release
of LAION-5B -- that is ~143k images the dataset maintainers withdrew. We verified this is
not link rot (87.6% of the URLs still resolve) and not a string-matching artifact. If you
need a set that tracks Re-LAION, intersect against the Re-LAION ids before use.
TXT
    tar -C "$STAGE" -cf - . | gcloud storage cp - "$OUT"
    echo "wrote $OUT"; exit 0
fi

case "$SET" in
    gt)      SRC="$LF/split_0";                  EXT=jpg; SUB=images  ;;
    teacher) SRC="$LF/teacher_recons_512_png";   EXT=png; SUB=teacher ;;
    *) echo "SET must be gt, teacher or meta"; exit 1 ;;
esac
DST="$BUCKET/split_0_1M_tar"
N=${NUM_SHARDS:-100}
K=$((SGE_TASK_ID - 1))
OUT="$DST/${SET}_$(printf %03d "$K")_of_$(printf %03d "$N").tar"

if gcloud storage ls "$OUT" >/dev/null 2>&1; then echo "exists, skipping: $OUT"; exit 0; fi

# every N-th file, so shards are equal-sized and the split is reproducible
LIST=$(mktemp); trap 'rm -f "$LIST"' EXIT
(cd "$SRC" && ls -1 | sort | awk -v k="$K" -v n="$N" 'NR%n==k+1 || (k+1==n && NR%n==0)') > "$LIST"
echo "$SET shard $K/$N: $(wc -l < "$LIST") files from $SRC"

tar -C "$SRC" -cf - --transform "s,^,$SUB/," -T "$LIST" | gcloud storage cp - "$OUT"
echo "wrote $OUT"
