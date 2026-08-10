"""Refresh, validate, and publish the production TLE catalog snapshot.

This module is the entrypoint for the scheduled Cloud Run Job. A failed
download or validation exits non-zero and leaves the versioned known-good
object in Cloud Storage untouched.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from backend.catalog import (
    catalog_epoch,
    download_catalog_snapshot,
    upload_catalog_snapshot,
    validate_catalog_file,
)
from training.data_download import RAW_DIR, build_catalog, seed_tle_cache_from_catalog


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    snapshot_uri = os.getenv("CATALOG_SNAPSHOT_URI", "").strip()
    if not snapshot_uri:
        raise RuntimeError("CATALOG_SNAPSHOT_URI is required")

    log.info("Starting scheduled catalog refresh for %s", snapshot_uri)
    catalog_path = Path(RAW_DIR) / "catalog.csv"
    download_catalog_snapshot(snapshot_uri, catalog_path)
    previous_epoch = catalog_epoch(catalog_path)
    seed_tle_cache_from_catalog(catalog_path)

    frame = build_catalog(use_cache=False)
    stats = validate_catalog_file(catalog_path)
    if len(frame) != stats["rows"]:
        raise RuntimeError("Catalog row count changed between build and validation")

    if stats["epoch"] <= previous_epoch:
        log.info(
            "Validated catalog did not advance beyond %s; keeping the current snapshot",
            previous_epoch.isoformat(),
        )
        return

    upload_catalog_snapshot(snapshot_uri, catalog_path)
    log.info(
        "Catalog refresh complete: %s rows, epoch %s, age %.2f hours",
        f"{stats['rows']:,}",
        stats["epoch"].isoformat(),
        stats["age_hours"],
    )


if __name__ == "__main__":
    main()
