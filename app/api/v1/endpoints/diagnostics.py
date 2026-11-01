from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse
from celery import chain
from celery.result import AsyncResult

from app.schemas.telemetry import TelemetryStreamBatch
from app.tasks.signal_processing import process_crucible_telemetry
from app.tasks.inference import run_tcn_diagnostics
from app.core.metrics import render_prometheus_metrics
from app.celery_app import celery_app

router = APIRouter()


@router.post(
    "/process-telemetry",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue High-Frequency Crucible Telemetry",
)
def enqueue_telemetry(payload: TelemetryStreamBatch):
    """Dispatches SVD filtering and 1D-TCN inference via a Celery Canvas chain."""
    telemetry_streams = {
        "Ts": payload.shell_temperature_Ts,
        "Bext": payload.magnetic_flux_Bext,
        "Vn": payload.neutral_voltage_Vn,
        "In": payload.leakage_current_In,
        "delta_Pc": payload.coolant_pressure_delta_Pc,
    }

    pipeline_chain = chain(
        process_crucible_telemetry.s(
            device_id=payload.device_id,
            telemetry_streams=telemetry_streams,
        ),
        run_tcn_diagnostics.s(),
    )
    task_async = pipeline_chain.apply_async()

    return {
        "status": "PENDING",
        "task_id": task_async.id,
        "device_id": payload.device_id,
        "message": "Telemetry stream successfully queued for diagnostics analysis.",
    }


@router.get(
    "/tasks/{task_id}",
    summary="Retrieve Diagnostics Task Status & Output",
)
def get_task_status(task_id: str):
    """Queries Redis result backend for pipeline execution state."""
    async_result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "status": async_result.status,
        "ready": async_result.ready(),
        "successful": async_result.successful(),
        "result": None,
    }

    if async_result.ready():
        if async_result.successful():
            response["result"] = async_result.result
        else:
            response["result"] = str(async_result.result)

    return response


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Retrieve Operational Engine & Inference Prometheus Metrics",
)
def read_diagnostics_metrics():
    """
    Exposes broker queue depth, worker inference latency quantiles,
    and cumulative failure mode counters in Prometheus exposition format.
    """
    return PlainTextResponse(
        content=render_prometheus_metrics(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )