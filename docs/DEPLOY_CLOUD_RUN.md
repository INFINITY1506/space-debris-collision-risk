# Deploy Debris Sentinel to Google Cloud Run

## Before deployment

1. Create or select a Google Cloud project with billing enabled.
2. Install the Google Cloud CLI and sign in with `gcloud auth login`.
3. Set a budget alert in Google Cloud Billing. A budget alerts you; it does not automatically stop resources.
4. Run `pytest -q`, `npm run build` in `frontend/`, and `npm audit`.
5. On projects that do not already grant it, give the default build account Google's documented Cloud Run Builder role:

   ```bash
   PROJECT_NUMBER="$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')"
   gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
     --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
     --role=roles/run.builder \
     --condition=None
   ```

## Deploy

From the repository root:

```bash
chmod +x scripts/deploy_cloud_run.sh
./scripts/deploy_cloud_run.sh YOUR_PROJECT_ID
```

The script enables Cloud Run, Cloud Build, Artifact Registry, Cloud Storage, and Cloud Scheduler. It then:

1. Creates a private, versioned catalog bucket with a lifecycle rule for older versions.
2. Creates separate least-privilege identities for the web service, refresh job, and scheduler.
3. Seeds the bucket only when no known-good snapshot exists.
4. Builds and deploys the public web service.
5. Deploys a private catalog refresh Cloud Run Job from the same immutable image.
6. Schedules the refresh daily at 02:15 UTC.
7. Checks `/health` after the new revision becomes ready.

The cost guardrails are scale-to-zero, concurrency 1, and a maximum of one web instance. A cold start downloads only the last validated Cloud Storage snapshot and loads the checkpoint; it never depends directly on CelesTrak.

The refresh job validates downloads before publishing. A failed or unchanged upstream response leaves the previous object generation untouched. CelesTrak's documented HTTP 403 "data has not updated" response is treated as unchanged data, not retried aggressively.

## Verify production

```bash
SERVICE_URL="$(gcloud run services describe debris-sentinel \
  --project YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --format='value(status.url)')"

curl "$SERVICE_URL/health"
curl -X POST "$SERVICE_URL/api/predict" \
  -H 'Content-Type: application/json' \
  -d '{"norad_id":25544,"top_n":3}'
```

Confirm that `screening_available` is `true`, the catalog is recent, the homepage loads, and the result says its ranking basis is `minimum propagated miss distance`. A health status of `degraded` means the catalog is older than the 48-hour target but still inside the 168-hour screening limit.

Inspect or run the refresh job manually:

```bash
gcloud run jobs execute debris-catalog-refresh \
  --project YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --wait

gcloud scheduler jobs describe debris-catalog-daily \
  --project YOUR_PROJECT_ID \
  --location asia-southeast1
```

## Connect debrissentinel.com

Cloud Run's direct domain mapping is currently a preview feature, but it is the simplest low-cost option for this portfolio project. The Singapore region supports it.

1. Verify ownership of the base domain:

   ```bash
   gcloud domains verify debrissentinel.com
   ```

2. Create the mapping:

   ```bash
   gcloud beta run domain-mappings create \
     --project YOUR_PROJECT_ID \
     --region asia-southeast1 \
     --service debris-sentinel \
     --domain debrissentinel.com
   ```

3. Retrieve the exact DNS records Google requires:

   ```bash
   gcloud beta run domain-mappings describe \
     --project YOUR_PROJECT_ID \
     --region asia-southeast1 \
     --domain debrissentinel.com
   ```

4. In Cloudflare DNS, create every returned record. Keep the records **DNS only** (grey cloud) until Google's certificate is active. Disable Cloudflare's **Always Use HTTPS** during certificate validation. Certificate provisioning normally takes around 15 minutes but can take up to 24 hours.

5. After HTTPS works, create a separate `www` mapping or redirect `www.debrissentinel.com` to the apex domain in Cloudflare.

For a larger commercial service, Google recommends an external Application Load Balancer instead of direct domain mapping, but that adds cost and complexity.

## Updates and rollback

Redeploy the current source with the same script. Cloud Run creates a new immutable revision and moves traffic only after deployment succeeds.

List revisions:

```bash
gcloud run revisions list \
  --project YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --service debris-sentinel
```

Route traffic to a known-good revision:

```bash
gcloud run services update-traffic debris-sentinel \
  --project YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --to-revisions REVISION_NAME=100
```

## Cost notes

Cloud Run's monthly free tier continues after the 90-day trial as long as billing remains enabled. With the configured scale-to-zero service, light portfolio traffic should often stay in the free tier. Artifact Registry, Cloud Storage, Cloud Scheduler, job execution, and outbound data can still generate small charges. The one-instance maximum limits simultaneous compute but is not a hard currency spending cap, so keep a billing budget alert enabled.
