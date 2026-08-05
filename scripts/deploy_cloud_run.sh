#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GOOGLE_CLOUD_PROJECT_ID" >&2
  exit 2
fi

PROJECT_ID="$1"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-debris-sentinel}"

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  --project "$PROJECT_ID"

gcloud run deploy "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --source . \
  --allow-unauthenticated \
  --execution-environment gen2 \
  --cpu 1 \
  --memory 2Gi \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 300 \
  --set-env-vars "REFRESH_TLE_ON_STARTUP=true,CATALOG_MAX_AGE_HOURS=48,ALLOW_STALE_CATALOG=false"

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

echo "Deployment complete: $SERVICE_URL"
curl --fail --silent --show-error \
  --http1.1 \
  --connect-timeout 10 \
  --max-time 20 \
  --retry 60 \
  --retry-delay 5 \
  --retry-all-errors \
  "$SERVICE_URL/health"
