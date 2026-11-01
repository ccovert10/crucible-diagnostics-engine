import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple


def kantz_schreiber_svd_filter(
    signal: np.ndarray,
    embedding_dim: int = 5,
    delay: int = 14,
    neighborhood_size: int = 30,
    subspace_dim: int = 2
) -> np.ndarray:
    """
    Kantz-Schreiber Local Projective SVD Noise Filter.
    Denoises non-linear time series in phase space without smearing sharp physical transients
    (e.g., induction coil micro-arcing spikes or localized thermal transients).

    Parameters:
    -----------
    signal : np.ndarray
        1D raw sensor array (e.g., harmonic leakage current In or shell temp Ts).
    embedding_dim : int
        Reconstructed phase space dimension (m). Default = 5.
    delay : int
        Time lag / delay reconstruction parameter (tau). Default = 14.
    neighborhood_size : int
        Number of nearest neighbors (k) used to construct local tangent manifolds. Default = 30.
    subspace_dim : int
        Number of dominant singular values (q) to retain on the attractor manifold. Default = 2.

    Returns:
    --------
    np.ndarray
        Cleaned 1D signal with preserved physical transients and suppressed stochastic noise.
    """
    x = np.asarray(signal, dtype=np.float64)
    N = len(x)

    # 1. Check window sufficiency for delay coordinate construction
    min_len = (embedding_dim - 1) * delay + 1
    if N < min_len:
        raise ValueError(
            f"Signal length {N} is too short for embedding_dim={embedding_dim} and delay={delay}. "
            f"Minimum required is {min_len} samples."
        )

    # 2. Construct the Delay Coordinate Trajectory Matrix: shape (M, embedding_dim)
    # Each row: [x[i], x[i + tau], x[i + 2*tau], ..., x[i + (m - 1)*tau]]
    M = N - (embedding_dim - 1) * delay
    trajectory_matrix = np.empty((M, embedding_dim), dtype=np.float64)
    for dim_idx in range(embedding_dim):
        start_idx = dim_idx * delay
        trajectory_matrix[:, dim_idx] = x[start_idx : start_idx + M]

    # Constrain neighborhood size k to available trajectory points
    k = min(neighborhood_size, M - 1)
    if k <= subspace_dim:
        return x.copy()

    # 3. Fast spatial neighbor search using a balanced k-d tree
    kdtree = cKDTree(trajectory_matrix)
    # Query k-nearest neighbors for all trajectory points simultaneously
    _, neighbor_indices = kdtree.query(trajectory_matrix, k=k)

    denoised_trajectories = np.empty_like(trajectory_matrix)

    # 4. Local Projective Subspace SVD filtering across each neighborhood
    for i in range(M):
        # Extract the k neighbor points: shape (k, embedding_dim)
        neighbors = trajectory_matrix[neighbor_indices[i]]

        # Compute centroid of local neighborhood
        centroid = np.mean(neighbors, axis=0)

        # Center the neighborhood matrix
        centered_matrix = neighbors - centroid

        # Compute SVD: centered_matrix = U * S * Vt
        # Vt has shape (embedding_dim, embedding_dim), rows are principal component axes
        _, _, Vt = np.linalg.svd(centered_matrix, full_matrices=False)

        # Retain only the top 'subspace_dim' principal components
        # V_q has shape (embedding_dim, subspace_dim)
        V_q = Vt[:subspace_dim, :].T

        # Projection operator onto local tangent space: P_q = V_q @ V_q.T
        point_centered = trajectory_matrix[i] - centroid
        projected_point = centroid + (point_centered @ V_q) @ V_q.T

        denoised_trajectories[i] = projected_point

    # 5. Diagonal Averaging (anti-diagonal reconstruction back to 1D)
    reconstructed_signal = np.zeros(N, dtype=np.float64)
    overlap_counts = np.zeros(N, dtype=np.float64)

    for dim_idx in range(embedding_dim):
        start_idx = dim_idx * delay
        reconstructed_signal[start_idx : start_idx + M] += denoised_trajectories[:, dim_idx]
        overlap_counts[start_idx : start_idx + M] += 1.0

    # Avoid division by zero on edges
    safe_counts = np.where(overlap_counts > 0, overlap_counts, 1.0)
    cleaned_signal = reconstructed_signal / safe_counts

    # Any boundary samples that weren't covered retain original values
    uncovered_mask = overlap_counts == 0
    if np.any(uncovered_mask):
        cleaned_signal[uncovered_mask] = x[uncovered_mask]

    return cleaned_signal