import os
import time
from typing import Dict, Any, List
import numpy as np
import redis

# Redis connection string with graceful fallback to Docker service default
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")

# Dedicated Redis client for operational telemetry
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

METRICS_KEY_PREFIX = "crucible:metrics"
LATENCY_LIST_KEY = f"{METRICS_KEY_PREFIX}:latencies_ms"
FAILURE_MODES_HASH = f"{METRICS_KEY_PREFIX}:failure_modes"
TOTAL_PROCESSED_KEY = f"{METRICS_KEY_PREFIX}:total_processed"
MAX_LATENCY_HISTORY = 1000  # Sliding window size of recorded inferences


def record_inference_telemetry(failure_mode: str, latency_ms: float) -> None:
    """
    Records an inference event:
    Increments total count, updates failure mode distribution,
    and appends execution latency to a rolling FIFO buffer.
    """
    pipe = redis_client.pipeline()
    pipe.incr(TOTAL_PROCESSED_KEY)
    pipe.hincrby(FAILURE_MODES_HASH, failure_mode, 1)
    pipe.lpush(LATENCY_LIST_KEY, latency_ms)
    pipe.ltrim(LATENCY_LIST_KEY, 0, MAX_LATENCY_HISTORY - 1)
    pipe.execute()


def get_diagnostics_metrics() -> Dict[str, Any]:
    """
    Aggregates operational latency distributions, Celery queue depth,
    and failure mode counts directly from Redis.
    """
    queue_depth = redis_client.llen("celery")
    total_processed = int(redis_client.get(TOTAL_PROCESSED_KEY) or 0)

    raw_modes = redis_client.hgetall(FAILURE_MODES_HASH)
    failure_mode_counts = {
        "CLASS_A_REFRACTORY_THINNING": int(raw_modes.get("CLASS_A_REFRACTORY_THINNING", 0)),
        "CLASS_B_INDUCTION_COIL_MICRO_ARCING": int(raw_modes.get("CLASS_B_INDUCTION_COIL_MICRO_ARCING", 0)),
        "CLASS_C_COOLING_CIRCUIT_BLOCKAGE": int(raw_modes.get("CLASS_C_COOLING_CIRCUIT_BLOCKAGE", 0)),
    }

    raw_latencies = redis_client.lrange(LATENCY_LIST_KEY, 0, -1)
    latencies = [float(val) for val in raw_latencies] if raw_latencies else []

    latency_stats = {
        "sample_count": len(latencies),
        "mean_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "min_ms": round(float(np.min(latencies)), 2) if latencies else 0.0,
        "p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else 0.0,
        "p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "p99_ms": round(float(np.percentile(latencies, 99)), 2) if latencies else 0.0,
        "max_ms": round(float(np.max(latencies)), 2) if latencies else 0.0,
    }

    return {
        "broker_queue_depth": queue_depth,
        "total_batches_processed": total_processed,
        "worker_latency_distribution": latency_stats,
        "failure_mode_counts": failure_mode_counts,
        "timestamp_utc": time.time(),
    }


def render_prometheus_metrics() -> str:
    """
    Renders diagnostic metrics into Prometheus / OpenMetrics exposition format.
    Compatible with Prometheus, Grafana Agent, and Datadog scrape agents.
    """
    metrics = get_diagnostics_metrics()
    latency = metrics["worker_latency_distribution"]
    failure_modes = metrics["failure_mode_counts"]

    lines: List[str] = [
        "# HELP crucible_broker_queue_depth Current pending task count in Celery Redis broker.",
        "# TYPE crucible_broker_queue_depth gauge",
        f"crucible_broker_queue_depth {metrics['broker_queue_depth']}",
        "",
        "# HELP crucible_batches_processed_total Total telemetry batches evaluated by 1D-TCN engine.",
        "# TYPE crucible_batches_processed_total counter",
        f"crucible_batches_processed_total {metrics['total_batches_processed']}",
        "",
        "# HELP crucible_worker_inference_latency_ms Worker inference latency quantiles over sliding window.",
        "# TYPE crucible_worker_inference_latency_ms summary",
        f'crucible_worker_inference_latency_ms{{quantile="0.5"}} {latency["p50_ms"]}',
        f'crucible_worker_inference_latency_ms{{quantile="0.95"}} {latency["p95_ms"]}',
        f'crucible_worker_inference_latency_ms{{quantile="0.99"}} {latency["p99_ms"]}',
        f"crucible_worker_inference_latency_ms_sum {round(latency['mean_ms'] * latency['sample_count'], 2)}",
        f"crucible_worker_inference_latency_ms_count {latency['sample_count']}",
        "",
        "# HELP crucible_failure_mode_verdict_total Inferred metallurgical failure mode occurrences.",
        "# TYPE crucible_failure_mode_verdict_total counter",
    ]

    for mode_name, count in failure_modes.items():
        lines.append(f'crucible_failure_mode_verdict_total{{class="{mode_name}"}} {count}')

    lines.append("")  # Final newline required by Prometheus exposition standard
    return "\n".join(lines)