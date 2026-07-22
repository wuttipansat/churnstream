import os
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27000")
os.environ.setdefault("MODEL_PATH", "artifacts/model.pkl")
os.environ.setdefault("MODEL_METADATA_PATH", "artifacts/model_metadata.json")

from churnstream.api import main, services

class FakeModel:
    def predict(self, input_data: Any) -> np.ndarray:
        return np.array([0])
    
    def predict_proba(self, input_data: Any) -> np.ndarray:
        return np.array([[0.67, 0.33]])
    
class FakeModelLoader:
    def __init__(self) -> None:
        self.model = FakeModel()

    def __call__(self) -> FakeModel:
        return self.model
    
    def cache_clear(self) -> None:
        pass

@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    fake_loader = FakeModelLoader()

    monkeypatch.setattr(main, "load_model", fake_loader)

    monkeypatch.setattr(services, "load_model", fake_loader)

    monkeypatch.setattr(main, "get_model_info", lambda: {"model_name": "logistic_regression", "model_version": "test",})

    with TestClient(main.app) as test_client:
        yield test_client