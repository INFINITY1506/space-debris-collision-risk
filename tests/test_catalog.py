from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.catalog import catalog_age_hours, catalog_epoch, ensure_catalog_fresh


def test_catalog_epoch_supports_two_digit_tle_years(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame(
        {"epoch_year": [26, 26, 26], "epoch_day": [100.0, 102.0, 104.0]}
    ).to_csv(path, index=False)

    assert catalog_epoch(path) == datetime(2026, 4, 12, tzinfo=timezone.utc)
    assert catalog_age_hours(path, datetime(2026, 4, 13, tzinfo=timezone.utc)) == 24.0


def test_stale_catalog_fails_closed_when_refresh_is_disabled(tmp_path):
    path = tmp_path / "catalog.csv"
    pd.DataFrame({"epoch_year": [20], "epoch_day": [1.0]}).to_csv(path, index=False)

    with pytest.raises(RuntimeError, match="Catalog is stale"):
        ensure_catalog_fresh(path, max_age_hours=48, refresh=False, allow_stale=False)
