import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from churnstream.api import services


class FakeModel:
    def predict(self, data: Any) -> list[int]:
        return np.array([0])

    def predict_proba(self, data: Any) -> list[list[float]]:
        return np.array([[0.7, 0.3]])


def test_load_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.pkl"
    model_path.touch()

    monkeypatch.setattr(
        services,
        "MODEL_PATH",
        model_path,
    )

    monkeypatch.setattr(
        services.joblib,
        "load",
        lambda path: FakeModel(),
    )

    services.load_model.cache_clear()

    try:
        model = services.load_model()

        assert isinstance(model, FakeModel)
    finally:
        services.load_model.cache_clear()


def test_load_model_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.pkl"

    monkeypatch.setattr(
        services,
        "MODEL_PATH",
        missing_path,
    )

    services.load_model.cache_clear()

    try:
        with pytest.raises(
            FileNotFoundError,
            match="Model file was not found",
        ):
            services.load_model()
    finally:
        services.load_model.cache_clear()


def test_get_model_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "model_metadata.json"

    metadata = {
        "model_name": "logistic_regression",
        "model_version": "1.0.0",
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        services,
        "MODEL_METADATA_PATH",
        metadata_path,
    )

    result = services.get_model_info()

    assert result == metadata


def test_get_model_info_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.json"

    monkeypatch.setattr(
        services,
        "MODEL_METADATA_PATH",
        missing_path,
    )

    result = services.get_model_info()

    assert result == {
        "model": "model.pkl",
        "metadata": "not available",
    }


def test_get_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from churnstream.schemas.data_schemas import PredictionInput

    monkeypatch.setattr(
        services,
        "load_model",
        lambda: FakeModel(),
    )

    customer = PredictionInput(
        CreditScore=650,
        Geography="France",
        Gender="Male",
        Age=42,
        Tenure=2,
        Balance=0.0,
        NumOfProducts=1,
        HasCrCard=1,
        IsActiveMember=1,
        EstimatedSalary=101348.88,
    )

    result = services.get_prediction(customer)

    assert result.prediction == 0
    assert result.probability == pytest.approx(0.3)