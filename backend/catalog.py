"""Catalog freshness checks used during application startup."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


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
