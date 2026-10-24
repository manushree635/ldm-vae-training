#!/bin/bash -l
# Download and unpack the 1M face set from the signed URLs into $LDM_DATA, then lay it out
# the way the 1M configs expect. See ../DATA.md for how to get the URL file.
#
#   LDM_DATA=/path/with/1TB bash scripts_rebuttal/download_1M_data.sh split_0_1M_urls.txt
#
#   URLS=...    file of signed URLs, one per line (201 of them: 100 gt + 100 teacher + meta)
#   JOBS=4      parallel transfers; raise only if your link, not the disk, is the limit
#   ONLY=all    all | teacher | meta -- "teacher" skips the 247 GB of source frames, which
#               you may already have or may prefer to re-fetch from the LAION URLs
#
# Streams each tar straight into tar -x, so the compressed copy is never written and the
# download needs only the unpacked size (~600 GB for everything, ~350 GB for teacher+meta).
# Re-running is safe: a shard whose files are already present is simply overwritten.
set -euo pipefail
URLS="${1:-${URLS:-split_0_1M_urls.txt}}"
: "${LDM_DATA:?set LDM_DATA to where the data should live (needs ~1 TB for the full set)}"
: "${JOBS:=4}"
: "${ONLY:=all}"
[ -f "$URLS" ] || { echo "no URL file at $URLS -- see DATA.md" >&2; exit 1; }

case "$ONLY" in
    all)     PAT='.' ;;
    teacher) PAT='/\(teacher_\|meta\)' ;;
    meta)    PAT='/meta' ;;
    *) echo "ONLY must be all, teacher or meta" >&2; exit 1 ;;
esac
SEL=$(grep "$PAT" "$URLS" || true)
N=$(printf '%s\n' "$SEL" | grep -c . || true)
[ "$N" -gt 0 ] || { echo "no URLs matched ONLY=$ONLY" >&2; exit 1; }

mkdir -p "$LDM_DATA"; cd "$LDM_DATA"
echo "unpacking $N shard(s) into $LDM_DATA with $JOBS parallel transfers"
# -f so curl fails loudly on an expired link instead of piping an HTML error page into tar
printf '%s\n' "$SEL" | xargs -P "$JOBS" -I{} sh -c 'curl -fsSL "{}" | tar -x'

echo
echo "unpacked:"
for d in images teacher split_0_1M; do
    [ -d "$d" ] && printf "  %-12s %s\n" "$d" "$(ls "$d" | wc -l)"
done
# the configs read these names; symlink rather than rename so a re-run stays idempotent
[ -d images  ] && ln -sfn "$LDM_DATA/images"  "$LDM_DATA/split_0"
[ -d teacher ] && ln -sfn "$LDM_DATA/teacher" "$LDM_DATA/teacher_recons_512_png"

echo
echo "expected 992574 in images/ and teacher/ for the full set."
echo "A short count means a shard failed -- re-run, it will refetch only what is missing."
echo "Next: bash scripts_rebuttal/prepare_data_1M.sh   (file lists + latent cache)"
