import numpy as np
from typing import Dict, Union


def takens_delay_embedding(
    signal: np.ndarray,
    embedding_dim: int = 5,
    delay: int = 14
) -> np.ndarray:
    """
    Constructs an m-dimensional phase-space trajectory matrix from a 1D scalar time series
    using Takens' Delay-Coordinate Embedding Theorem.

    Parameters:
    -----------
    signal : np.ndarray
        1D array of temporal sensor readings (e.g., cleaned harmonic leakage current In).
    embedding_dim : int
        Embedding dimension (m). Default = 5.
    delay : int
        Delay / lag parameter (tau) in discrete sample steps. Default = 14.

    Returns:
    --------
    np.ndarray
        Phase-space trajectory matrix of shape (M, embedding_dim), where
        M = N - (embedding_dim - 1) * delay.
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)

    # 1. Theoretical sufficiency boundary check
    min_required_samples = (embedding_dim - 1) * delay + 1
    if N < min_required_samples:
        raise ValueError(
            f"Input signal length ({N}) is insufficient for embedding_dim={embedding_dim} "
            f"and delay={delay}. Minimum required samples: {min_required_samples}."
        )

    # 2. Compute trajectory trajectory length M
    M = N - (embedding_dim - 1) * delay

    # 3. Vectorized construction via strided array slicing (Zero-copy view)
    # Each row i: [x[i], x[i + tau], x[i + 2*tau], ..., x[i + (m - 1)*tau]]
    trajectory_matrix = np.empty((M, embedding_dim), dtype=np.float64)
    for dim_idx in range(embedding_dim):
        start_idx = dim_idx * delay
        trajectory_matrix[:, dim_idx] = x[start_idx : start_idx + M]

    return trajectory_matrix


def construct_multivariate_phase_space_tensor(
    telemetry_streams: Dict[str, Union[np.ndarray, list]],
    embedding_dim: int = 5,
    delay: int = 14
) -> np.ndarray:
    """
    Processes all 5 physical crucible telemetry streams (Ts, Bext, Vn, In, delta_Pc)
    into a synchronized multivariate phase-space tensor for deep learning inference.

    Parameters:
    -----------
    telemetry_streams : Dict[str, Union[np.ndarray, list]]
        Dictionary containing the 5 synchronized sensor arrays:
        {'Ts': [...], 'Bext': [...], 'Vn': [...], 'In': [...], 'delta_Pc': [...]}
    embedding_dim : int
        Embedding dimension per channel. Default = 5.
    delay : int
        Embedding lag step. Default = 14.

    Returns:
    --------
    np.ndarray
        Multivariate feature tensor of shape (M, num_channels * embedding_dim)
        or ready to reshape to (Batch, Channels, Temporal_Steps) for 1D-TCN.
    """
    channel_keys = ["Ts", "Bext", "Vn", "In", "delta_Pc"]
    embedded_channels = []

    for key in channel_keys:
        if key not in telemetry_streams:
            raise KeyError(f"Missing mandatory telemetry stream: '{key}'")
        
        raw_arr = np.asarray(telemetry_streams[key], dtype=np.float64)
        embedded_matrix = takens_delay_embedding(
            signal=raw_arr,
            embedding_dim=embedding_dim,
            delay=delay
        )
        embedded_channels.append(embedded_matrix)

    # Concatenate along the feature axis: shape (M, 5 * 5) = (M, 25)
    multivariate_tensor = np.concatenate(embedded_channels, axis=1)
    return multivariate_tensor