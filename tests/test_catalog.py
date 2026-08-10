from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.catalog import (
    catalog_age_hours,
    catalog_epoch,
    download_catalog_snapshot,
    ensure_catalog_fresh,
    parse_gs_uri,
    upload_catalog_snapshot,
    validate_catalog_file,
)


def _valid_catalog(day: float) -> pd.DataFrame:
    sources = ["active", "cosmos", "fengyun", "iridium"]
    return pd.DataFrame(
        {
            "name": [f"OBJECT {i}" for i in range(4)],
            "norad_id": [100 + i for i in range(4)],
            "epoch_year": [26] * 4,
            "epoch_day": [day] * 4,
            "source": sources,
            "line1": [f"1 {i}" for i in range(4)],
            "line2": [f"2 {i}" for i in range(4)],
        }
    )


class _FakeBlob:
    def __init__(self, source):
        self.source = source
        self.metadata = None
        self.uploaded = None

    def download_to_filename(self, destination):
        destination_path = type(self.source)(destination)
        destination_path.write_bytes(self.source.read_bytes())

    def upload_from_filename(self, source, content_type=None):
        self.uploaded = (source, content_type)


class _FakeBucket:
    def __init__(self, blob):
        self._blob = blob

    def blob(self, _name):
        return self._blob


class _FakeClient:
    def __init__(self, blob):
        self._blob = blob

    def bucket(self, _name):
        return _FakeBucket(self._blob)


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


def test_parse_gs_uri_requires_bucket_and_object():
    assert parse_gs_uri("gs://catalog-bucket/catalog/current.csv") == (
        "catalog-bucket",
        "catalog/current.csv",
    )
    with pytest.raises(ValueError):
        parse_gs_uri("https://example.com/catalog.csv")


def test_snapshot_download_is_validated_and_installed_atomically(tmp_path):
    local = tmp_path / "catalog.csv"
    remote = tmp_path / "remote.csv"
    _valid_catalog(100.0).to_csv(local, index=False)
    _valid_catalog(101.0).to_csv(remote, index=False)

    changed = download_catalog_snapshot(
        "gs://catalog-bucket/catalog/current.csv",
        local,
        client=_FakeClient(_FakeBlob(remote)),
        min_rows=4,
    )

    assert changed is True
    assert catalog_epoch(local) == datetime(2026, 4, 11, tzinfo=timezone.utc)
    assert not local.with_suffix(".csv.snapshot.tmp").exists()


def test_invalid_snapshot_never_replaces_known_good_catalog(tmp_path):
    local = tmp_path / "catalog.csv"
    remote = tmp_path / "remote.csv"
    _valid_catalog(100.0).to_csv(local, index=False)
    pd.DataFrame({"norad_id": [1]}).to_csv(remote, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        download_catalog_snapshot(
            "gs://catalog-bucket/catalog/current.csv",
            local,
            client=_FakeClient(_FakeBlob(remote)),
            min_rows=1,
        )

    assert catalog_epoch(local) == datetime(2026, 4, 10, tzinfo=timezone.utc)


def test_upload_publishes_only_a_validated_catalog(tmp_path):
    path = tmp_path / "catalog.csv"
    _valid_catalog(101.0).to_csv(path, index=False)
    blob = _FakeBlob(path)

    stats = upload_catalog_snapshot(
        "gs://catalog-bucket/catalog/current.csv",
        path,
        client=_FakeClient(blob),
        min_rows=4,
    )

    assert stats["rows"] == 4
    assert blob.uploaded == (str(path), "text/csv")
    assert blob.metadata["catalog_rows"] == "4"
