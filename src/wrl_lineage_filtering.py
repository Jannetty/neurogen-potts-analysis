from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.expdata_geo_helpers import (
    assign_cells_to_lineages_by_containment,
    lineage_is_connected,
    load_vrml_meshes,
    to_trimesh_list,
)


@dataclass(frozen=True)
class WrlLineageFilterConfig:
    min_lineage_faces: int = 50
    max_secondary_volume: float = 5.0
    min_pros_fraction: float = 0.95
    min_dpn_fraction: float = 0.60


@dataclass(frozen=True)
class FilteredLineage:
    lineage_index: int
    lineage_mesh: Any
    pros_indices: np.ndarray
    dpn_indices: np.ndarray

    @property
    def n_pros(self) -> int:
        return int(self.pros_indices.size)

    @property
    def n_dpn(self) -> int:
        return int(self.dpn_indices.size)


@dataclass(frozen=True)
class RejectedLineage:
    lineage_index: int
    lineage_mesh: Any
    rejection_reason: str   # "disconnected" | "no_dpn"
    pros_indices: np.ndarray
    dpn_indices: np.ndarray

    @property
    def n_pros(self) -> int:
        return int(self.pros_indices.size)

    @property
    def n_dpn(self) -> int:
        return int(self.dpn_indices.size)


@dataclass(frozen=True)
class FilteredWrlLobe:
    lineage_meshes: list[Any]
    pros_meshes: list[Any]
    dpn_meshes: list[Any]
    pros_to_lineage_idx: np.ndarray
    dpn_to_lineage_idx: np.ndarray
    kept_lineages: list[FilteredLineage]
    rejected_lineages: list[RejectedLineage]
    stats: dict[str, int]


def load_filtered_wrl_lobe(
    lineage_file: Path,
    pros_file: Path,
    dpn_file: Path,
    *,
    config: WrlLineageFilterConfig | None = None,
) -> FilteredWrlLobe:
    if config is None:
        config = WrlLineageFilterConfig()

    lineage_meshes = to_trimesh_list(load_vrml_meshes(lineage_file))
    pros_meshes = to_trimesh_list(load_vrml_meshes(pros_file))
    dpn_meshes = to_trimesh_list(load_vrml_meshes(dpn_file))

    pros_to_lineage_idx, _ = assign_cells_to_lineages_by_containment(
        pros_meshes, lineage_meshes, min_fraction=config.min_pros_fraction
    )
    dpn_to_lineage_idx, _ = assign_cells_to_lineages_by_containment(
        dpn_meshes, lineage_meshes, min_fraction=config.min_dpn_fraction
    )

    kept_lineages: list[FilteredLineage] = []
    rejected_lineages: list[RejectedLineage] = []
    stats = {
        "lineages_total": len(lineage_meshes),
        "lineages_kept": 0,
        "lineages_skipped_disconnected": 0,
        "lineages_skipped_no_dpn": 0,
    }

    for lineage_index, lineage_mesh in enumerate(lineage_meshes):
        dpn_indices = np.where(dpn_to_lineage_idx == lineage_index)[0]
        pros_indices = np.where(pros_to_lineage_idx == lineage_index)[0]

        if not lineage_is_connected(lineage_mesh, min_faces=config.min_lineage_faces, max_secondary_volume=config.max_secondary_volume):
            stats["lineages_skipped_disconnected"] += 1
            rejected_lineages.append(
                RejectedLineage(
                    lineage_index=lineage_index,
                    lineage_mesh=lineage_mesh,
                    rejection_reason="disconnected",
                    pros_indices=pros_indices,
                    dpn_indices=dpn_indices,
                )
            )
            continue

        if dpn_indices.size == 0:
            stats["lineages_skipped_no_dpn"] += 1
            rejected_lineages.append(
                RejectedLineage(
                    lineage_index=lineage_index,
                    lineage_mesh=lineage_mesh,
                    rejection_reason="no_dpn",
                    pros_indices=pros_indices,
                    dpn_indices=dpn_indices,
                )
            )
            continue

        kept_lineages.append(
            FilteredLineage(
                lineage_index=lineage_index,
                lineage_mesh=lineage_mesh,
                pros_indices=pros_indices,
                dpn_indices=dpn_indices,
            )
        )
        stats["lineages_kept"] += 1

    return FilteredWrlLobe(
        lineage_meshes=lineage_meshes,
        pros_meshes=pros_meshes,
        dpn_meshes=dpn_meshes,
        pros_to_lineage_idx=pros_to_lineage_idx,
        dpn_to_lineage_idx=dpn_to_lineage_idx,
        kept_lineages=kept_lineages,
        rejected_lineages=rejected_lineages,
        stats=stats,
    )
