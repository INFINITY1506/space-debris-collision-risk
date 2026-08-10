#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GOOGLE_CLOUD_PROJECT_ID" >&2
  exit 2
fi

PROJECT_ID="$1"
REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-debris-sentinel}"
REFRESH_JOB="${REFRESH_JOB:-debris-catalog-refresh}"
SCHEDULER_JOB="${SCHEDULER_JOB:-debris-catalog-daily}"
BUCKET="${CATALOG_BUCKET:-${PROJECT_ID}-debris-sentinel-catalog}"
SNAPSHOT_URI="gs://${BUCKET}/catalog/current.csv"
RUNTIME_SA_NAME="debris-sentinel-runtime"
REFRESH_SA_NAME="debris-catalog-refresh"
SCHEDULER_SA_NAME="debris-catalog-scheduler"
RUNTIME_SA="${RUNTIME_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REFRESH_SA="${REFRESH_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
SCHEDULER_SA="${SCHEDULER_SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

ensure_service_account() {
  local account_name="$1"
  local display_name="$2"
  local account_email="${account_name}@${PROJECT_ID}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$account_email" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$account_name" \
      --project "$PROJECT_ID" \
      --display-name "$display_name"
  fi
}

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  storage.googleapis.com \
  --project "$PROJECT_ID"

ensure_service_account "$RUNTIME_SA_NAME" "Debris Sentinel runtime"
ensure_service_account "$REFRESH_SA_NAME" "Debris catalog refresh job"
ensure_service_account "$SCHEDULER_SA_NAME" "Debris catalog scheduler"

if ! gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention
fi

gcloud storage buckets update "gs://${BUCKET}" \
  --project "$PROJECT_ID" \
  --versioning \
  --lifecycle-file scripts/catalog_lifecycle.json

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project "$PROJECT_ID" \
  --member "serviceAccount:${RUNTIME_SA}" \
  --role roles/storage.objectViewer

gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --project "$PROJECT_ID" \
  --member "serviceAccount:${REFRESH_SA}" \
  --role roles/storage.objectAdmin

if ! gcloud storage ls "$SNAPSHOT_URI" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage cp data/raw/catalog.csv "$SNAPSHOT_URI" --project "$PROJECT_ID"
fi

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
  --service-account "$RUNTIME_SA" \
  --set-env-vars "CATALOG_SNAPSHOT_URI=${SNAPSHOT_URI},SYNC_CATALOG_SNAPSHOT_ON_STARTUP=true,REFRESH_TLE_ON_STARTUP=false,CATALOG_MAX_AGE_HOURS=48,CATALOG_HARD_MAX_AGE_HOURS=168,ALLOW_STALE_CATALOG=true"

IMAGE="$(gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(spec.template.spec.containers[0].image)')"

gcloud run jobs deploy "$REFRESH_JOB" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --image "$IMAGE" \
  --command python \
  --args=-m,training.catalog_refresh_job \
  --service-account "$REFRESH_SA" \
  --set-env-vars "CATALOG_SNAPSHOT_URI=${SNAPSHOT_URI}" \
  --cpu 1 \
  --memory 1Gi \
  --tasks 1 \
  --max-retries 0 \
  --task-timeout 15m

gcloud run jobs add-iam-policy-binding "$REFRESH_JOB" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --member "serviceAccount:${SCHEDULER_SA}" \
  --role roles/run.invoker

SCHEDULER_URI="https://run.googleapis.com/v2/projects/${PROJECT_ID}/locations/${REGION}/jobs/${REFRESH_JOB}:run"
if gcloud scheduler jobs describe "$SCHEDULER_JOB" --project "$PROJECT_ID" --location "$REGION" >/dev/null 2>&1; then
  gcloud scheduler jobs update http "$SCHEDULER_JOB" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --schedule "15 2 * * *" \
    --time-zone "Etc/UTC" \
    --uri "$SCHEDULER_URI" \
    --http-method POST \
    --oauth-service-account-email "$SCHEDULER_SA" \
    --message-body '{}' \
    --attempt-deadline 180s \
    --max-retry-attempts 2 \
    --min-backoff 300s
else
  gcloud scheduler jobs create http "$SCHEDULER_JOB" \
    --project "$PROJECT_ID" \
    --location "$REGION" \
    --schedule "15 2 * * *" \
    --time-zone "Etc/UTC" \
    --uri "$SCHEDULER_URI" \
    --http-method POST \
    --oauth-service-account-email "$SCHEDULER_SA" \
    --message-body '{}' \
    --attempt-deadline 180s \
    --max-retry-attempts 2 \
    --min-backoff 300s
fi

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format='value(status.url)')"

echo "Deployment complete: $SERVICE_URL"
echo "Catalog snapshot: $SNAPSHOT_URI"
echo "Catalog refresh schedule: 02:15 UTC daily"
curl --fail --silent --show-error \
  --http1.1 \
  --connect-timeout 10 \
  --max-time 20 \
  --retry 60 \
  --retry-delay 5 \
  --retry-all-errors \
  "$SERVICE_URL/health"
