import httpx
import pytest

BASE_URL = "http://localhost:8000/api/v1"


def test_health_check():
    """Verify application health and Celery broker connection."""
    response = httpx.get("http://localhost:8000/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["broker_configured"] is True


def test_get_diagnostics_history_unfiltered():
    """Verify pagination and default query against PostgreSQL."""
    response = httpx.get(f"{BASE_URL}/diagnostics/history?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "total_records" in data
    assert "records" in data
    assert isinstance(data["records"], list)
    assert len(data["records"]) <= 5


def test_get_diagnostics_history_filtered():
    """Verify exact filtering by device_id and failure_mode."""
    device_id = "CRUCIBLE-003"
    failure_mode = "CLASS_C_COOLING_CIRCUIT_BLOCKAGE"
    response = httpx.get(
        f"{BASE_URL}/diagnostics/history",
        params={
            "device_id": device_id,
            "failure_mode": failure_mode,
            "limit": 5,
        },
    )
    assert response.status_code == 200
    data = response.json()
    for record in data["records"]:
        assert record["device_id"] == device_id
        assert record["predicted_failure_mode"] == failure_mode
        assert "class_probabilities" in record
        assert "confidence" in record