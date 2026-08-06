import pandas as pd
import pytest

from backend.predictor import SatellitePredictor, risk_level_from_miss_distance


def test_candidate_selection_excludes_active_spacecraft_and_primary():
    predictor = SatellitePredictor.__new__(SatellitePredictor)
    predictor.catalog = pd.DataFrame(
        [
            {"norad_id": 25544, "name": "ISS", "source": "active"},
            {"norad_id": 48274, "name": "ISS MODULE", "source": "active"},
            {"norad_id": 33757, "name": "COSMOS DEBRIS", "source": "cosmos"},
            {"norad_id": 33545, "name": "FENGYUN DEBRIS", "source": "fengyun"},
        ]
    )

    result = predictor._select_debris_candidates(25544)

    assert result["norad_id"].tolist() == [33757, 33545]


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(0.999, "HIGH"), (1.0, "MEDIUM"), (4.999, "MEDIUM"), (5.0, "LOW")],
)
def test_risk_thresholds_are_explicit(distance, expected):
    assert risk_level_from_miss_distance(distance) == expected


def test_missing_checkpoint_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        SatellitePredictor(
            model_path=tmp_path / "missing.pth",
            catalog_path=tmp_path / "catalog.csv",
            norm_path=tmp_path / "normalization.npz",
            device="cpu",
        )


def test_catalog_listing_can_select_active_or_debris_objects():
    predictor = SatellitePredictor.__new__(SatellitePredictor)
    predictor.catalog = pd.DataFrame(
        [
            {"norad_id": 1, "name": "ACTIVE ONE", "source": "active", "altitude_km": 400, "inclination_deg": 51},
            {"norad_id": 2, "name": "ACTIVE TWO", "source": "active", "altitude_km": 500, "inclination_deg": 52},
            {"norad_id": 3, "name": "COSMOS DEBRIS", "source": "cosmos", "altitude_km": 600, "inclination_deg": 53},
            {"norad_id": 4, "name": "FENGYUN DEBRIS", "source": "fengyun", "altitude_km": 700, "inclination_deg": 54},
        ]
    )

    active = predictor.list_satellites(limit=10, kind="active")
    debris = predictor.list_satellites(limit=10, kind="debris")

    assert [item["norad_id"] for item in active] == [1, 2]
    assert [item["norad_id"] for item in debris] == [3, 4]
