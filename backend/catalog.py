"""Catalog validation, freshness checks, and durable snapshot storage."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "name",
    "norad_id",
    "epoch_year",
    "epoch_day",
    "source",
    "line1",
    "line2",
}
REQUIRED_SOURCES = {"active", "cosmos", "fengyun", "iridium"}


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """Split a ``gs://bucket/object`` URI into its bucket and object names."""
    if not uri.startswith("gs://"):
        raise ValueError("Catalog snapshot URI must start with gs://")
    bucket, separator, object_name = uri[5:].partition("/")
    if not bucket or not separator or not object_name:
        raise ValueError("Catalog snapshot URI must include a bucket and object name")
    return bucket, object_name


def validate_catalog_file(
    path: str | Path,
    *,
    min_rows: int = 1_000,
    required_sources: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a catalog before it is allowed to replace a known-good copy."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    frame = pd.read_csv(catalog_path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Catalog is missing columns: {sorted(missing_columns)}")
    if len(frame) < min_rows:
        raise ValueError(f"Catalog contains only {len(frame):,} rows; expected at least {min_rows:,}")
    if frame["norad_id"].duplicated().any():
        raise ValueError("Catalog contains duplicate NORAD IDs")

    expected_sources = REQUIRED_SOURCES if required_sources is None else required_sources
    missing_sources = expected_sources.difference(set(frame["source"].dropna().astype(str)))
    if missing_sources:
        raise ValueError(f"Catalog is missing required sources: {sorted(missing_sources)}")

    epoch = catalog_epoch(catalog_path)
    return {
        "rows": len(frame),
        "epoch": epoch,
        "age_hours": catalog_age_hours(catalog_path),
        "sources": sorted(set(frame["source"].dropna().astype(str))),
    }


def _storage_client(client: Any = None):
    if client is not None:
        return client
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("google-cloud-storage is required for catalog snapshots") from exc
    return storage.Client()


def download_catalog_snapshot(
    uri: str,
    destination: str | Path,
    *,
    client: Any = None,
    min_rows: int = 1_000,
) -> bool:
    """Atomically install a validated Cloud Storage snapshot when it is newer.

    Returns ``True`` when the destination was replaced and ``False`` when the
    bundled/local catalog was already at least as recent.
    """
    bucket_name, object_name = parse_gs_uri(uri)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination_path.with_suffix(destination_path.suffix + ".snapshot.tmp")

    storage_client = _storage_client(client)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    try:
        blob.download_to_filename(str(temp_path))
        remote = validate_catalog_file(temp_path, min_rows=min_rows)
        if destination_path.exists():
            try:
                local_epoch = catalog_epoch(destination_path)
            except (ValueError, FileNotFoundError):
                local_epoch = datetime.min.replace(tzinfo=timezone.utc)
            if remote["epoch"] <= local_epoch:
                log.info("Bundled catalog is as recent as snapshot %s", uri)
                return False
        os.replace(temp_path, destination_path)
        log.info(
            "Installed catalog snapshot %s (%s rows, epoch %s)",
            uri,
            f"{remote['rows']:,}",
            remote["epoch"].isoformat(),
        )
        return True
    finally:
        temp_path.unlink(missing_ok=True)


def upload_catalog_snapshot(
    uri: str,
    source: str | Path,
    *,
    client: Any = None,
    min_rows: int = 1_000,
) -> dict[str, Any]:
    """Validate and upload a catalog as the new versioned known-good snapshot."""
    stats = validate_catalog_file(source, min_rows=min_rows)
    bucket_name, object_name = parse_gs_uri(uri)
    storage_client = _storage_client(client)
    blob = storage_client.bucket(bucket_name).blob(object_name)
    blob.metadata = {
        "catalog_epoch": stats["epoch"].isoformat(),
        "catalog_rows": str(stats["rows"]),
        "catalog_sources": ",".join(stats["sources"]),
    }
    blob.upload_from_filename(str(source), content_type="text/csv")
    log.info("Published validated catalog snapshot %s (%s rows)", uri, f"{stats['rows']:,}")
    return stats


def catalog_epoch(path: str | Path) -> datetime:
    """Return the median TLE epoch in a catalog as an aware UTC datetime."""
    catalog_path = Path(path)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    frame = pd.read_csv(catalog_path, usecols=["epoch_year", "epoch_day"])
    if frame.empty:
        raise ValueError("Catalog is empty")

    years = frame["epoch_year"].astype(int).to_numpy()
    years = np.where(years < 100, np.where(years < 57, years + 2000, years + 1900), years)
    days = frame["epoch_day"].astype(float).to_numpy()

    timestamps = [
        datetime(int(year), 1, 1, tzinfo=timezone.utc) + timedelta(days=float(day) - 1)
        for year, day in zip(years, days)
    ]
    timestamps.sort()
    return timestamps[len(timestamps) // 2]


def catalog_age_hours(path: str | Path, now: datetime | None = None) -> float:
    """Return catalog age in hours using its median object epoch."""
    current = now or datetime.now(tz=timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current - catalog_epoch(path)).total_seconds() / 3600.0)


def ensure_catalog_fresh(
    path: str | Path,
    max_age_hours: float = 48.0,
    refresh: bool = True,
    allow_stale: bool = False,
) -> float:
    """Refresh a stale catalog and return its final age in hours.

    A bundled catalog is retained if a network refresh fails. Production can
    reject that fallback by leaving ``allow_stale`` disabled.
    """
    catalog_path = Path(path)
    try:
        age = catalog_age_hours(catalog_path)
    except (FileNotFoundError, ValueError):
        age = float("inf")

    if refresh and age > max_age_hours:
        log.info("Catalog age is %.1f hours; downloading a fresh TLE snapshot", age)
        try:
            from training.data_download import build_catalog

            build_catalog(use_cache=False)
        except Exception:
            log.exception("Catalog refresh failed")
        age = catalog_age_hours(catalog_path)

    if age > max_age_hours and not allow_stale:
        raise RuntimeError(
            f"Catalog is stale ({age:.1f} hours old; maximum is {max_age_hours:.1f})"
        )

    return age
