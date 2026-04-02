from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.expdata_geo_helpers import (
    assign_cells_to_lineages_strict,
    lineage_is_connected,
    load_vrml_meshes,
    mesh_bbox,
    to_trimesh_list,
)


@dataclass(frozen=True)
class WrlLineageFilterConfig:
    min_lineage_faces: int = 50
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
class FilteredWrlLobe:
    lineage_meshes: list[Any]
    pros_meshes: list[Any]
    dpn_meshes: list[Any]
    pros_to_lineage_idx: np.ndarray
    dpn_to_lineage_idx: np.ndarray
    kept_lineages: list[FilteredLineage]
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

    lineage_bboxes = [mesh_bbox(mesh) for mesh in lineage_meshes]
    pros_to_lineage_idx, _ = assign_cells_to_lineages_strict(
        pros_meshes, lineage_bboxes, min_fraction=config.min_pros_fraction
    )
    dpn_to_lineage_idx, _ = assign_cells_to_lineages_strict(
        dpn_meshes, lineage_bboxes, min_fraction=config.min_dpn_fraction
    )

    kept_lineages: list[FilteredLineage] = []
    stats = {
        "lineages_total": len(lineage_meshes),
        "lineages_kept": 0,
        "lineages_skipped_disconnected": 0,
        "lineages_skipped_no_dpn": 0,
    }

    for lineage_index, lineage_mesh in enumerate(lineage_meshes):
        if not lineage_is_connected(lineage_mesh, min_faces=config.min_lineage_faces):
            stats["lineages_skipped_disconnected"] += 1
            continue

        dpn_indices = np.where(dpn_to_lineage_idx == lineage_index)[0]
        if dpn_indices.size == 0:
            stats["lineages_skipped_no_dpn"] += 1
            continue

        pros_indices = np.where(pros_to_lineage_idx == lineage_index)[0]
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
        stats=stats,
    )
