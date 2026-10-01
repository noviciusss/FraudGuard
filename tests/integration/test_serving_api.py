"""Integration tests for Phase 2: FastAPI serving endpoints, schema validation, and idempotency."""

import pytest
from fastapi.testclient import TestClient

from fraudguard.persistence.database import init_db
from fraudguard.serving.app import app, load_model_and_config


@pytest.fixture(scope="module")
def client():
    """Initializes database and returns FastAPI test client."""
    init_db("sqlite:///test_serving.db")
    load_model_and_config()
    with TestClient(app) as test_client:
        yield test_client


def test_health_live_endpoint(client: TestClient):
    """Verify liveness probe returns status ok."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_ready_endpoint(client: TestClient):
    """Verify readiness probe verifies model loading and database connectivity."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_model_info_endpoint(client: TestClient):
    """Verify model info returns immutable version, threshold, and feature policy."""
    resp = client.get("/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "model_version" in data
    assert "decision_threshold" in data
    assert "oldbalanceOrg" in data["forbidden_columns"]


def test_predict_success_and_idempotency(client: TestClient):
    """Verify inference returns valid score, logs to database, and is strictly idempotent."""
    payload = {
        "transaction_id": "tx-test-idempotent-001",
        "event_step": 105,
        "type": "TRANSFER",
        "amount": 150000.0,
        "schema_version": "v1",
        "replay_run_id": "test_run_1",
    }

    # 1. First request
    resp1 = client.post("/predict", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert "prediction_id" in data1
    assert "fraud_score" in data1
    assert "decision" in data1
    assert data1["transaction_id"] == "tx-test-idempotent-001"
    assert 0.0 <= data1["fraud_score"] <= 1.0

    # 2. Second request with identical transaction_id and replay_run_id (Idempotency check)
    resp2 = client.post("/predict", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.json()

    # Must return exact same prediction record and ID
    assert data1["prediction_id"] == data2["prediction_id"]
    assert data1["fraud_score"] == data2["fraud_score"]
    assert data1["decision"] == data2["decision"]


def test_predict_rejects_forbidden_columns(client: TestClient):
    """Verify that sending forbidden balance columns or target labels fails schema validation (HTTP 422)."""
    payload_with_forbidden = {
        "transaction_id": "tx-forbidden-002",
        "event_step": 105,
        "type": "CASH_OUT",
        "amount": 2500.0,
        "oldbalanceOrg": 50000.0,  # FORBIDDEN column!
    }

    resp = client.post("/predict", json=payload_with_forbidden)
    # Pydantic extra='forbid' must reject this request
    assert resp.status_code == 422
    assert "extra_forbidden" in str(resp.json()) or "Extra inputs are not permitted" in str(resp.json())


def test_predict_rejects_negative_amount(client: TestClient):
    """Verify negative transaction amounts are rejected."""
    payload = {
        "transaction_id": "tx-invalid-amount",
        "event_step": 10,
        "type": "PAYMENT",
        "amount": -50.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
