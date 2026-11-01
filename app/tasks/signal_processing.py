import numpy as np
from celery import shared_task
from typing import Dict, Any, List
from app.engine.signal.svd_filter import kantz_schreiber_svd_filter
from app.engine.signal.phase_space import construct_multivariate_phase_space_tensor


@shared_task(
    bind=True,
    name="tasks.signal_processing.process_crucible_telemetry",
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3
)
def process_crucible_telemetry(
    self,
    device_id: str,
    telemetry_streams: Dict[str, List[float]],
    embedding_dim: int = 5,
    delay: int = 14
) -> Dict[str, Any]:
    """
    Distributed Celery Task:
    1. Denoises all 5 sensor channels using Kantz-Schreiber SVD filtering.
    2. Reconstructs the 25-feature multivariate phase-space trajectory tensor.
    3. Packages the tensor and telemetry metadata for downstream inference chaining.
    """
    denoised_streams = {}
    noise_reduction_stats = {}

    for channel_name, raw_values in telemetry_streams.items():
        raw_arr = np.asarray(raw_values, dtype=np.float64)
        
        filtered_arr = kantz_schreiber_svd_filter(
            signal=raw_arr,
            embedding_dim=embedding_dim,
            delay=delay,
            neighborhood_size=20,
            subspace_dim=2
        )
        
        raw_var = float(np.var(raw_arr))
        filtered_var = float(np.var(filtered_arr))
        
        denoised_streams[channel_name] = filtered_arr.tolist()
        noise_reduction_stats[channel_name] = {
            "raw_variance": round(raw_var, 4),
            "filtered_variance": round(filtered_var, 4),
            "variance_suppression_ratio": round(filtered_var / (raw_var + 1e-9), 4)
        }

    phase_space_tensor = construct_multivariate_phase_space_tensor(
        telemetry_streams=denoised_streams,
        embedding_dim=embedding_dim,
        delay=delay
    )

    return {
        "device_id": device_id,
        "signal_task_id": self.request.id,
        "trajectory_points": phase_space_tensor.shape[0],
        "feature_dimensions": phase_space_tensor.shape[1],
        "noise_stats": noise_reduction_stats,
        "phase_space_centroid": np.mean(phase_space_tensor, axis=0).tolist(),
        # Pass the 2D matrix (M, 25) downstream to the inference task
        "phase_space_tensor": phase_space_tensor.tolist()
    }