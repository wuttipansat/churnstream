import pytest
from fastapi.testclient import TestClient

VALID_CUSTOMER = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 2,
    "Balance": 0.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
}

def test_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "ChurnStream API",
        "docs": "/docs",
    }

def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "loaded",
    }

def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "logistic_regression",
        "model_version": "test"
    }

def test_predict_success(client: TestClient) -> None:
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "probability": pytest.approx(0.33),
    }

def test_predict_missing_field(client: TestClient) -> None:
    payload = VALID_CUSTOMER.copy()
    payload.pop("Age")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("Age", 17),
        ("Balance", -1),
        ("Geography", "Thailand"),
        ("Gender", "Other"),
        ("HasCrCard", 2),
        ("IsActiveMember", 2),
    ],
)

def test_predict_reject_invalid_values(
    client: TestClient,
    field: str,
    invalid_value: object,
) -> None:
    payload = VALID_CUSTOMER.copy()
    payload[field] = invalid_value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

def test_predict_rejects_extra_field(
        client: TestClient
) -> None:
    payload = {
        **VALID_CUSTOMER,
        "UnknownFeature": 100,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422