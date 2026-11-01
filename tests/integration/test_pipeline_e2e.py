import time
from typing import Any, Dict
import httpx
import pytest


@pytest.fixture(scope="session")
def http_client():
    with httpx.Client(base_url="http://localhost:8000", timeout=10.0) as client:
        yield client


def generate_synthetic_telemetry(n_samples: int = 100) -> Dict[str, Any]:
    """Generates synthetic multi-channel crucible telemetry matching TelemetryStreamBatch schema."""
    import numpy as np

    np.random.seed(42)
    return {
        "device_id": "CRUCIBLE-BF-TEST",
        "timestamp": "2026-10-27T12:00:00Z",
        "sampling_rate_hz": 100,
        "shell_temperature_Ts": (1420.0 + np.random.randn(n_samples) * 8.5).tolist(),
        "magnetic_flux_Bext": (11.5 + np.random.randn(n_samples) * 0.9).tolist(),
        "neutral_voltage_Vn": (0.6 + np.random.randn(n_samples) * 0.12).tolist(),
        "leakage_current_In": (2.2 + np.random.randn(n_samples) * 0.3).tolist(),
        "coolant_pressure_delta_Pc": (0.45 + np.random.randn(n_samples) * 0.1).tolist(),
    }


def test_api_gateway_health(http_client: httpx.Client):
    """Ensure the FastAPI gateway is reachable and healthy."""
    response = http_client.get("/docs")
    assert response.status_code == 200


def test_schema_validation_rejection(http_client: httpx.Client):
    """Verify that omitting a mandatory field (e.g. timestamp) triggers HTTP 422."""
    payload = generate_synthetic_telemetry()
    del payload["timestamp"]

    response = http_client.post("/api/v1/diagnostics/process-telemetry", json=payload)
    assert response.status_code == 422


def test_end_to_end_diagnostics_pipeline(http_client: httpx.Client):
    """
    Submits a valid telemetry stream, polls the Redis result backend,
    and asserts the validity of the computed analytical output.
    """
    payload = generate_synthetic_telemetry(n_samples=100)

    # 1. Enqueue task through API Gateway
    post_res = http_client.post("/api/v1/diagnostics/process-telemetry", json=payload)
    assert post_res.status_code == 202, f"Failed to enqueue telemetry batch: {post_res.text}"

    envelope = post_res.json()
    task_id = envelope["task_id"]
    assert task_id is not None
    assert envelope["status"] == "PENDING"
    assert envelope["device_id"] == "CRUCIBLE-BF-TEST"

    # 2. Poll Redis backend via task status endpoint
    max_retries = 20
    poll_interval = 0.5
    final_result = None

    for _ in range(max_retries):
        status_res = http_client.get(f"/api/v1/diagnostics/tasks/{task_id}")
        assert status_res.status_code == 200
        status_data = status_res.json()

        if status_data.get("ready"):
            assert status_data["successful"] is True, f"Pipeline execution failed: {status_data}"
            final_result = status_data["result"]
            break
        time.sleep(poll_interval)

    assert final_result is not None, f"Pipeline timed out after {max_retries * poll_interval} seconds."

    # 3. Model output domain invariants
    assert final_result["device_id"] == "CRUCIBLE-BF-TEST"
    assert final_result["status"] == "COMPLETED"
    assert final_result["sequence_steps_evaluated"] == 44
    assert final_result["predicted_failure_mode"] in [
        "CLASS_A_REFRACTORY_THINNING",
        "CLASS_B_INDUCTION_COIL_MICRO_ARCING",
        "CLASS_C_COOLING_CIRCUIT_BLOCKAGE",
    ]
    assert 0.0 <= final_result["confidence"] <= 1.0
    assert final_result["lead_time_to_failure_hours"] >= 0.0

    # 4. Audit provenance checks
    assert "signal_task_id" in final_result
    assert "inference_task_id" in final_result
    assert final_result["inference_task_id"] == task_id


def test_diagnostics_metrics_endpoint(http_client: httpx.Client):
    """Verify that operational metrics are exported in Prometheus format."""
    res = http_client.get("/api/v1/diagnostics/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]

    body = res.text
    # Verify metric families and type definitions
    assert "# TYPE crucible_broker_queue_depth gauge" in body
    assert "# TYPE crucible_batches_processed_total counter" in body
    assert "# TYPE crucible_worker_inference_latency_ms summary" in body
    assert "# TYPE crucible_failure_mode_verdict_total counter" in body

    # Verify metric values and labels
    assert "crucible_broker_queue_depth " in body
    assert "crucible_batches_processed_total " in body
    assert 'crucible_failure_mode_verdict_total{class="CLASS_A_REFRACTORY_THINNING"}' in body
    assert 'crucible_failure_mode_verdict_total{class="CLASS_B_INDUCTION_COIL_MICRO_ARCING"}' in body
    assert 'crucible_failure_mode_verdict_total{class="CLASS_C_COOLING_CIRCUIT_BLOCKAGE"}' in body