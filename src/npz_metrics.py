from __future__ import annotations

import numpy as np


def mean_std_n(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def compute_sizes_from_geo(geo: np.ndarray) -> dict[str, np.ndarray]:
    """
    geo: (N, H, W, 2)
      channel 0 = dpn mask
      channel 1 = union mask for pops {2, 3} (may overlap dpn)

    Returns per-lineage voxel-count proxies: dpn_sizes, pros_only_sizes, lineage_sizes.
    """
    assert geo.ndim == 4 and geo.shape[-1] == 2, f"Unexpected geo shape: {geo.shape}"
    N = geo.shape[0]

    dpn = geo[..., 0] > 0.5
    union_raw = geo[..., 1] > 0.5
    pros_only = union_raw & (~dpn)
    lineage = dpn | pros_only

    return {
        "dpn_sizes": dpn.reshape(N, -1).sum(axis=1).astype(np.float64),
        "pros_only_sizes": pros_only.reshape(N, -1).sum(axis=1).astype(np.float64),
        "lineage_sizes": lineage.reshape(N, -1).sum(axis=1).astype(np.float64),
    }


def compute_metrics(geo: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """
    counts: (N, ≥2) where column 0 = dpn cell count, column 1 = pros-like cell count.
    If a third column is present it is used as the sum of per-NB voxel counts (exp data).
    """
    assert counts.ndim == 2 and counts.shape[1] >= 2, f"Unexpected counts shape: {counts.shape}"

    sizes = compute_sizes_from_geo(geo)
    dpn_counts = counts[:, 0].astype(np.float64)
    pros_counts = counts[:, 1].astype(np.float64)
    total_counts = dpn_counts + pros_counts
    lineage_volumes = sizes["lineage_sizes"]

    # Use per-NB voxel sum when available (exp NPZs); fall back to union mask for sim.
    dpn_sum_voxels = counts[:, 2].astype(np.float64) if counts.shape[1] >= 3 else sizes["dpn_sizes"]
    avg_nb_volume = dpn_sum_voxels / np.maximum(dpn_counts, 1.0)

    return {
        "cell_counts_total": total_counts,
        "lineage_volumes": lineage_volumes,
        "avg_nb_volume": avg_nb_volume,
        "avg_lineage_volume": lineage_volumes,
    }
