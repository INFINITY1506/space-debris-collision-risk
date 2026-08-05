import asyncio

from fastapi.responses import JSONResponse

import backend.main as main


def test_health_returns_503_until_predictor_is_ready(monkeypatch):
    monkeypatch.setattr(main, "predictor", None)
    monkeypatch.setattr(main, "predictor_state", "loading")
    monkeypatch.setattr(main, "predictor_error", None)

    response = asyncio.run(main.health_check())

    assert isinstance(response, JSONResponse)
    assert response.status_code == 503
