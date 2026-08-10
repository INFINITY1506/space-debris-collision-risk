import asyncio
import json

import pytest
from fastapi.responses import JSONResponse

import backend.main as main


class _LoadedPredictor:
    loaded = True
    catalog = [object()]
    device = "cpu"


def test_health_returns_503_until_predictor_is_ready(monkeypatch):
    monkeypatch.setattr(main, "predictor", None)
    monkeypatch.setattr(main, "predictor_state", "loading")
    monkeypatch.setattr(main, "predictor_error", None)

    response = asyncio.run(main.health_check())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "5"
    payload = json.loads(response.body)
    assert payload["status"] == "loading"
    assert payload["detail"].startswith("Service is warming up")


def test_health_reports_aging_catalog_without_taking_site_offline(monkeypatch):
    monkeypatch.setattr(main, "predictor", _LoadedPredictor())
    monkeypatch.setattr(main, "predictor_state", "ready")
    monkeypatch.setattr(main, "loaded_catalog_age_hours", 72.0)
    monkeypatch.setattr(main, "catalog_screening_available", True)

    response = asyncio.run(main.health_check())

    assert response.status == "degraded"
    assert response.catalog_state == "aging"
    assert response.screening_available is True
    assert response.detail.startswith("Catalog refresh is delayed")


def test_hard_stale_catalog_pauses_screening(monkeypatch):
    monkeypatch.setattr(main, "predictor", _LoadedPredictor())
    monkeypatch.setattr(main, "loaded_catalog_age_hours", 200.0)
    monkeypatch.setattr(main, "catalog_screening_available", False)

    with pytest.raises(main.HTTPException) as exc_info:
        main._require_screening_predictor()

    assert exc_info.value.status_code == 503
    assert "screening is paused" in exc_info.value.detail
