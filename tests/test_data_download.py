import pandas as pd
import requests

import training.data_download as data_download


def test_unchanged_403_reuses_validated_source_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(data_download, "RAW_DIR", tmp_path)
    cached = "OBJECT\n1 00001U\n2 00001\n"
    (tmp_path / "active.tle").write_text(cached)

    response = requests.Response()
    response.status_code = 403
    response.url = "https://celestrak.org/active"
    response._content = (
        b"GP data has not updated since your last successful\n"
        b"download of GROUP=active."
    )
    monkeypatch.setattr(data_download.requests, "get", lambda *args, **kwargs: response)

    result = data_download.download_tle("active", response.url, retries=1)

    assert result == cached


def test_catalog_snapshot_seeds_per_source_tle_cache(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr(data_download, "RAW_DIR", raw_dir)
    catalog = tmp_path / "catalog.csv"
    pd.DataFrame(
        {
            "name": ["ACTIVE OBJECT", "DEBRIS OBJECT"],
            "line1": ["1 ACTIVE", "1 DEBRIS"],
            "line2": ["2 ACTIVE", "2 DEBRIS"],
            "source": ["active", "cosmos"],
        }
    ).to_csv(catalog, index=False)

    data_download.seed_tle_cache_from_catalog(catalog)

    assert (raw_dir / "active.tle").read_text() == "ACTIVE OBJECT\n1 ACTIVE\n2 ACTIVE\n"
    assert (raw_dir / "cosmos.tle").read_text() == "DEBRIS OBJECT\n1 DEBRIS\n2 DEBRIS\n"
    assert not (raw_dir / "starlink.tle").exists()
