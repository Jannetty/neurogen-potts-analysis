"""
preprocess_lineages_for_viz.py

Runs the same WRL filtering pipeline used by the analysis scripts, then
saves each kept lineage as a compact NPZ cache file so the visualizer can
load any lineage instantly without re-parsing all WRL files.

Outputs
-------
data/exp/viz_cache/lineage_index.csv
    One row per kept lineage with columns:
        lineage_id, genotype, lobe, lineage_idx, n_dpn, n_pros,
        total_cells, volume_um3

data/exp/viz_cache/meshes/<lineage_id>.npz
    Per-lineage mesh data:
        lin_vertices, lin_faces                      – lineage hull
        dpn_<i>_vertices, dpn_<i>_faces  (i=0,1,…)  – Dpn neuroblast meshes
        pros_<i>_vertices, pros_<i>_faces (i=0,1,…)  – Pros cell meshes

data/exp/viz_cache/rejected_lineage_index.csv
    One row per rejected lineage with same columns plus:
        rejection_reason  – "disconnected" | "no_dpn" | "wt_multi_dpn"

data/exp/viz_cache/rejected_meshes/<lineage_id>.npz
    Same format as meshes/ above.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh

from src.wrl_lineage_filtering import WrlLineageFilterConfig, load_filtered_wrl_lobe


# ---------------------------------------------------------------------------
# WRL file discovery  (mirrors plot_exp_wrl_histograms.py)
# ---------------------------------------------------------------------------

def find_lobe_triplets(
    base_dir: Path, genotype: str
) -> List[Tuple[str, Path, Path, Path]]:
    if genotype == "wt":
        sub = base_dir / "Control"
        lineage_suffix = "_control_lineages.wrl"
        dpn_suffix = "_control_dpn.wrl"
        pros_suffix = "_control_pros.wrl"
    elif genotype == "mudmut":
        sub = base_dir / "Mud"
        lineage_suffix = "_mud_lineages.wrl"
        dpn_suffix = "_mud_dpn.wrl"
        pros_suffix = "_mud_pros.wrl"
    else:
        raise ValueError(f"Unsupported genotype: {genotype}")

    lineage_files = sorted(sub.glob(f"*{lineage_suffix}"))
    if not lineage_files:
        raise FileNotFoundError(f"No lineage WRL files in {sub}")

    triplets: List[Tuple[str, Path, Path, Path]] = []
    for lf in lineage_files:
        lobe_name = lf.name.replace(lineage_suffix, "")
        dpn_file = sub / f"{lobe_name}{dpn_suffix}"
        pros_file = sub / f"{lobe_name}{pros_suffix}"
        if not dpn_file.exists():
            raise FileNotFoundError(f"Missing DPN file: {dpn_file}")
        if not pros_file.exists():
            raise FileNotFoundError(f"Missing Pros file: {pros_file}")
        triplets.append((lobe_name, lf, dpn_file, pros_file))
    return triplets


# ---------------------------------------------------------------------------
# Volume helper
# ---------------------------------------------------------------------------

def mesh_volume_um3(mesh: trimesh.Trimesh) -> float:
    vol = abs(float(mesh.volume)) / 1000
    return vol if math.isfinite(vol) else float("nan")


# ---------------------------------------------------------------------------
# Cache writing
# ---------------------------------------------------------------------------

def save_lineage_npz(
    path: Path,
    lineage_mesh: trimesh.Trimesh,
    dpn_meshes: List[trimesh.Trimesh],
    pros_meshes: List[trimesh.Trimesh],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "lin_vertices": np.asarray(lineage_mesh.vertices, dtype=np.float32),
        "lin_faces": np.asarray(lineage_mesh.faces, dtype=np.int32),
    }
    for i, m in enumerate(dpn_meshes):
        arrays[f"dpn_{i}_vertices"] = np.asarray(m.vertices, dtype=np.float32)
        arrays[f"dpn_{i}_faces"] = np.asarray(m.faces, dtype=np.int32)
    for i, m in enumerate(pros_meshes):
        arrays[f"pros_{i}_vertices"] = np.asarray(m.vertices, dtype=np.float32)
        arrays[f"pros_{i}_faces"] = np.asarray(m.faces, dtype=np.int32)
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache filtered WRL lineage meshes for the interactive visualizer."
    )
    parser.add_argument(
        "--wrl-dir",
        type=Path,
        default=Path("data/exp/wrl_files"),
        help="Base directory containing Control/ and Mud/ sub-dirs (default: data/exp/wrl_files)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/exp/viz_cache"),
        help="Output directory for cache files (default: data/exp/viz_cache)",
    )
    parser.add_argument(
        "--min-lineage-faces",
        type=int,
        default=50,
        help="Min faces for a secondary component to be considered significant (default: 50)",
    )
    parser.add_argument(
        "--max-secondary-volume",
        type=float,
        default=5.0,
        help="Max volume of a secondary component before the lineage is rejected as disconnected (default: 5.0)",
    )
    parser.add_argument(
        "--dpn-min-fraction",
        type=float,
        default=0.60,
        help="Min bbox overlap fraction for Dpn assignment (default: 0.60)",
    )
    parser.add_argument(
        "--pros-min-fraction",
        type=float,
        default=0.95,
        help="Min bbox overlap fraction for Pros assignment (default: 0.95)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = WrlLineageFilterConfig(
        min_lineage_faces=args.min_lineage_faces,
        max_secondary_volume=args.max_secondary_volume,
        min_pros_fraction=args.pros_min_fraction,
        min_dpn_fraction=args.dpn_min_fraction,
    )

    mesh_dir = args.out_dir / "meshes"
    rejected_mesh_dir = args.out_dir / "rejected_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    rejected_mesh_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.out_dir / "lineage_index.csv"
    rejected_index_path = args.out_dir / "rejected_lineage_index.csv"

    index_rows: list[dict] = []
    rejected_rows: list[dict] = []
    lineage_id_counter = 0
    rejected_id_counter = 0

    for genotype in ("wt", "mudmut"):
        try:
            triplets = find_lobe_triplets(args.wrl_dir, genotype)
        except FileNotFoundError as exc:
            print(f"  Skipping {genotype}: {exc}")
            continue

        for lobe_name, lineage_file, dpn_file, pros_file in triplets:
            print(f"[{genotype}] {lobe_name}: parsing...", end=" ", flush=True)
            filtered_lobe = load_filtered_wrl_lobe(
                lineage_file, pros_file, dpn_file, config=config
            )
            print(
                f"kept {filtered_lobe.stats['lineages_kept']}/"
                f"{filtered_lobe.stats['lineages_total']} lineages"
            )

            for fl in filtered_lobe.kept_lineages:
                dpn_meshes = [filtered_lobe.dpn_meshes[i] for i in fl.dpn_indices]
                pros_meshes = [filtered_lobe.pros_meshes[i] for i in fl.pros_indices]

                # WT: single-Dpn only — multi-Dpn go to rejected
                if genotype == "wt" and fl.n_dpn != 1:
                    rejected_id = f"rej{rejected_id_counter:04d}"
                    rejected_id_counter += 1
                    save_lineage_npz(
                        rejected_mesh_dir / f"{rejected_id}.npz",
                        fl.lineage_mesh,
                        dpn_meshes,
                        pros_meshes,
                    )
                    volume = mesh_volume_um3(fl.lineage_mesh)
                    rejected_rows.append(
                        {
                            "lineage_id": rejected_id,
                            "genotype": genotype,
                            "lobe": lobe_name,
                            "lineage_idx": fl.lineage_index,
                            "n_dpn": fl.n_dpn,
                            "n_pros": fl.n_pros,
                            "total_cells": fl.n_dpn + fl.n_pros,
                            "volume_um3": f"{volume:.2f}" if math.isfinite(volume) else "nan",
                            "rejection_reason": "wt_multi_dpn",
                        }
                    )
                    continue

                lineage_id = f"lin{lineage_id_counter:04d}"
                lineage_id_counter += 1
                save_lineage_npz(
                    mesh_dir / f"{lineage_id}.npz",
                    fl.lineage_mesh,
                    dpn_meshes,
                    pros_meshes,
                )
                volume = mesh_volume_um3(fl.lineage_mesh)
                index_rows.append(
                    {
                        "lineage_id": lineage_id,
                        "genotype": genotype,
                        "lobe": lobe_name,
                        "lineage_idx": fl.lineage_index,
                        "n_dpn": fl.n_dpn,
                        "n_pros": fl.n_pros,
                        "total_cells": fl.n_dpn + fl.n_pros,
                        "volume_um3": f"{volume:.2f}" if math.isfinite(volume) else "nan",
                    }
                )

            for rl in filtered_lobe.rejected_lineages:
                rejected_id = f"rej{rejected_id_counter:04d}"
                rejected_id_counter += 1
                dpn_meshes = [filtered_lobe.dpn_meshes[i] for i in rl.dpn_indices]
                pros_meshes = [filtered_lobe.pros_meshes[i] for i in rl.pros_indices]
                save_lineage_npz(
                    rejected_mesh_dir / f"{rejected_id}.npz",
                    rl.lineage_mesh,
                    dpn_meshes,
                    pros_meshes,
                )
                volume = mesh_volume_um3(rl.lineage_mesh)
                rejected_rows.append(
                    {
                        "lineage_id": rejected_id,
                        "genotype": genotype,
                        "lobe": lobe_name,
                        "lineage_idx": rl.lineage_index,
                        "n_dpn": rl.n_dpn,
                        "n_pros": rl.n_pros,
                        "total_cells": rl.n_dpn + rl.n_pros,
                        "volume_um3": f"{volume:.2f}" if math.isfinite(volume) else "nan",
                        "rejection_reason": rl.rejection_reason,
                    }
                )

    kept_fieldnames = [
        "lineage_id", "genotype", "lobe", "lineage_idx",
        "n_dpn", "n_pros", "total_cells", "volume_um3",
    ]
    with index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kept_fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    rejected_fieldnames = kept_fieldnames + ["rejection_reason"]
    with rejected_index_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rejected_fieldnames)
        writer.writeheader()
        writer.writerows(rejected_rows)

    print(f"\nCached {len(index_rows)} kept lineages, {len(rejected_rows)} rejected lineages.")
    print(f"  Index:          {index_path}")
    print(f"  Rejected index: {rejected_index_path}")
    print(f"  Meshes:         {mesh_dir}/")
    print(f"  Rejected meshes:{rejected_mesh_dir}/")


if __name__ == "__main__":
    main()
