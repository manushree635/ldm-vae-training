#!/bin/bash -l
# 7-day signed download URLs for the 1M shards (gt + teacher + meta).
# Run on a LOGIN node -- it needs your own gcloud credentials, not a compute node's:
#   ./make_1M_signed_urls.sh [out_file]
#
# 7 days is the maximum gcloud will sign for. To cut access sooner, delete the objects
# or rotate the service-account key; a signed URL cannot be revoked individually.
#
# One-time prerequisite (lets YOUR account sign as the service account):
#   gcloud iam service-accounts add-iam-policy-binding "$SA" \
#     --member=user:<you>@gmail.com --role=roles/iam.serviceAccountTokenCreator
set -euo pipefail
SA=${GCS_SIGNER_SA:?set GCS_SIGNER_SA to the service account allowed to sign}
PREFIX=${GCS_PREFIX:-gs://vae-distill/split_0_1M_tar}
OUT=${1:-split_0_1M_urls.txt}
module load python3/3.10.12 google-cloud-sdk/455.0.0
: > "$OUT"
for f in $(gcloud storage ls "$PREFIX/"); do
  gcloud storage sign-url "$f" --duration=7d \
    --impersonate-service-account="$SA" --format="value(signed_url)" >> "$OUT"
done
echo "wrote $(wc -l < "$OUT") urls to $OUT  (expire $(date -d '+7 days' '+%Y-%m-%d'))"
echo "Treat this file as a credential: anyone holding a link can download, no sign-in."
