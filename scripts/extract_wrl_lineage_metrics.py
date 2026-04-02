from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.expdata_geo_helpers import project_lineage_polygon_2d
from src.npz_metrics import mean_std_n
from src.wrl_lineage_filtering import WrlLineageFilterConfig, load_filtered_wrl_lobe

CONDITIONS = {
    "wt": {
        "label": "WT",
        "base_dir": Path("data/exp/wrl_files/Control"),
        "lobes": (1, 2, 7, 8, 13, 15),
        "lineage_template": "lobe{lobe}_control_lineages.wrl",
        "pros_template": "lobe{lobe}_control_pros.wrl",
        "dpn_template": "lobe{lobe}_control_dpn.wrl",
    },
    "mudmut": {
        "label": "mudmut",
        "base_dir": Path("data/exp/wrl_files/Mud"),
        "lobes": (110, 113, 116, 119),
        "lineage_template": "lobe{lobe}_mud_lineages.wrl",
        "pros_template": "lobe{lobe}_mud_pros.wrl",
        "dpn_template": "lobe{lobe}_mud_dpn.wrl",
    },
    "nanobody": {
        "label": "nanobody",
        "base_dir": Path("data/exp/wrl_files/Nanobody"),
        "lobes": (16, 17, 19, 20, 21, 22),
        "lineage_template": "lobe{lobe}_Nanobody_lineages.wrl",
        "pros_template": "lobe{lobe}_Nanobody_pros.wrl",
        "dpn_template": "lobe{lobe}_Nanobody_dpn.wrl",
    },
}


@dataclass
class LineageRecord:
    condition: str
    lobe: int
    lineage_index: int
    n_dpn: int
    n_pros: int
    total_cell_count: int
    lineage_volume_um3: float
    lineage_projected_area_um2: float
    avg_neuroblast_volume_um3: float
    avg_neuroblast_projected_area_um2: float


def safe_mesh_volume_um3(mesh) -> float:
    vol = float(mesh.volume)
    if not np.isfinite(vol):
        return float("nan")
    return abs(vol)


def lineage_projected_area_um2(mesh) -> float:
    return mesh_projected_area_um2(mesh)


def mesh_projected_area_um2(mesh) -> float:
    vertices = mesh.vertices
    if len(vertices) < 3:
        return float("nan")
    mean_v = vertices.mean(axis=0)
    centered = vertices - mean_v
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    poly = project_lineage_polygon_2d(
        lineage_mesh_L=mesh,
        plane_origin=mean_v,
        u_axis=e1,
        v_axis=e2,
        buffer_radius=0.0,
    )
    if poly is None or poly.is_empty:
        return float("nan")
    return float(poly.area)
def analyze_lobe(
    condition_key: str,
    lobe: int,
    min_lineage_faces: int = 50,
    min_pros_fraction: float = 0.95,
    min_dpn_fraction: float = 0.60,
) -> tuple[list[LineageRecord], dict[str, int]]:
    cfg = CONDITIONS[condition_key]
    base_dir = cfg["base_dir"]

    lineage_file = base_dir / cfg["lineage_template"].format(lobe=lobe)
    pros_file = base_dir / cfg["pros_template"].format(lobe=lobe)
    dpn_file = base_dir / cfg["dpn_template"].format(lobe=lobe)

    filtered_lobe = load_filtered_wrl_lobe(
        lineage_file,
        pros_file,
        dpn_file,
        config=WrlLineageFilterConfig(
            min_lineage_faces=min_lineage_faces,
            min_pros_fraction=min_pros_fraction,
            min_dpn_fraction=min_dpn_fraction,
        ),
    )
    lineage_tm = filtered_lobe.lineage_meshes
    pros_tm = filtered_lobe.pros_meshes
    dpn_tm = filtered_lobe.dpn_meshes

    records: list[LineageRecord] = []
    stats = dict(filtered_lobe.stats)

    for filtered_lineage in filtered_lobe.kept_lineages:
        lineage_index = filtered_lineage.lineage_index
        lineage_mesh = filtered_lineage.lineage_mesh
        dpn_meshes = [dpn_tm[i] for i in filtered_lineage.dpn_indices]
        n_dpn = filtered_lineage.n_dpn
        n_pros = filtered_lineage.n_pros
        total_cell_count = n_dpn + n_pros

        lineage_volume = safe_mesh_volume_um3(lineage_mesh)
        proj_area = lineage_projected_area_um2(lineage_mesh)

        dpn_volumes = np.asarray(
            [safe_mesh_volume_um3(mesh) for mesh in dpn_meshes],
            dtype=float,
        )
        dpn_projected_areas = np.asarray(
            [mesh_projected_area_um2(mesh) for mesh in dpn_meshes],
            dtype=float,
        )
        avg_nb_volume = float(np.nanmean(dpn_volumes))
        avg_nb_projected_area = float(np.nanmean(dpn_projected_areas))

        records.append(
            LineageRecord(
                condition=cfg["label"],
                lobe=lobe,
                lineage_index=lineage_index,
                n_dpn=n_dpn,
                n_pros=n_pros,
                total_cell_count=total_cell_count,
                lineage_volume_um3=lineage_volume,
                lineage_projected_area_um2=proj_area,
                avg_neuroblast_volume_um3=avg_nb_volume,
                avg_neuroblast_projected_area_um2=avg_nb_projected_area,
            )
        )

    return records, stats


def summarize_condition(
    records: list[LineageRecord], stats: dict[str, int]
) -> dict[str, object]:
    return {
        "condition": records[0].condition if records else "",
        "lineages_total": stats["lineages_total"],
        "lineages_kept": stats["lineages_kept"],
        "lineages_skipped_disconnected": stats["lineages_skipped_disconnected"],
        "lineages_skipped_no_dpn": stats["lineages_skipped_no_dpn"],
        "mean_lineage_volume_um3": mean_std_n(r.lineage_volume_um3 for r in records)[0],
        "std_lineage_volume_um3": mean_std_n(r.lineage_volume_um3 for r in records)[1],
        "mean_lineage_projected_area_um2": mean_std_n(
            r.lineage_projected_area_um2 for r in records
        )[0],
        "std_lineage_projected_area_um2": mean_std_n(
            r.lineage_projected_area_um2 for r in records
        )[1],
        "mean_avg_neuroblast_volume_um3": mean_std_n(
            r.avg_neuroblast_volume_um3 for r in records
        )[0],
        "std_avg_neuroblast_volume_um3": mean_std_n(
            r.avg_neuroblast_volume_um3 for r in records
        )[1],
        "mean_avg_neuroblast_projected_area_um2": mean_std_n(
            r.avg_neuroblast_projected_area_um2 for r in records
        )[0],
        "std_avg_neuroblast_projected_area_um2": mean_std_n(
            r.avg_neuroblast_projected_area_um2 for r in records
        )[1],
        "mean_total_cell_count": mean_std_n(r.total_cell_count for r in records)[0],
        "std_total_cell_count": mean_std_n(r.total_cell_count for r in records)[1],
        "n_lineages_in_summary": len(records),
    }


def write_per_lineage_csv(path: Path, records: list[LineageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "lobe",
        "lineage_index",
        "n_dpn",
        "n_pros",
        "total_cell_count",
        "lineage_volume_um3",
        "lineage_projected_area_um2",
        "avg_neuroblast_volume_um3",
        "avg_neuroblast_projected_area_um2",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "lineages_total",
        "lineages_kept",
        "lineages_skipped_disconnected",
        "lineages_skipped_no_dpn",
        "mean_lineage_volume_um3",
        "std_lineage_volume_um3",
        "mean_lineage_projected_area_um2",
        "std_lineage_projected_area_um2",
        "mean_avg_neuroblast_volume_um3",
        "std_avg_neuroblast_volume_um3",
        "mean_avg_neuroblast_projected_area_um2",
        "std_avg_neuroblast_projected_area_um2",
        "mean_total_cell_count",
        "std_total_cell_count",
        "n_lineages_in_summary",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def print_condition_summary(summary: dict[str, object]) -> None:
    print(f"\n=== {summary['condition']} ===")
    print(
        "Lineages:"
        f" total={summary['lineages_total']}"
        f" kept={summary['lineages_kept']}"
        f" skipped_disconnected={summary['lineages_skipped_disconnected']}"
        f" skipped_no_dpn={summary['lineages_skipped_no_dpn']}"
    )
    print(
        "Average lineage volume in 3D (um^3):"
        f" mean={summary['mean_lineage_volume_um3']:.3f}"
        f" std={summary['std_lineage_volume_um3']:.3f}"
        f" N={summary['n_lineages_in_summary']}"
    )
    print(
        "Average lineage projected area in 2D (um^2):"
        f" mean={summary['mean_lineage_projected_area_um2']:.3f}"
        f" std={summary['std_lineage_projected_area_um2']:.3f}"
        f" N={summary['n_lineages_in_summary']}"
    )
    print(
        "Average neuroblast volume (um^3):"
        f" mean={summary['mean_avg_neuroblast_volume_um3']:.3f}"
        f" std={summary['std_avg_neuroblast_volume_um3']:.3f}"
        f" N={summary['n_lineages_in_summary']}"
    )
    print(
        "Average neuroblast projected area in 2D (um^2):"
        f" mean={summary['mean_avg_neuroblast_projected_area_um2']:.3f}"
        f" std={summary['std_avg_neuroblast_projected_area_um2']:.3f}"
        f" N={summary['n_lineages_in_summary']}"
    )
    print(
        "Average cell count:"
        f" mean={summary['mean_total_cell_count']:.3f}"
        f" std={summary['std_total_cell_count']:.3f}"
        f" N={summary['n_lineages_in_summary']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute WRL-based experimental lineage metrics directly from the "
            "original meshes before NPZ conversion."
        )
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/exp/wrl_metrics"),
        help="Directory for summary CSV outputs.",
    )
    parser.add_argument(
        "--min-lineage-faces",
        type=int,
        default=50,
        help="Minimum face count for connected components in lineage filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_records: list[LineageRecord] = []
    all_summaries: list[dict[str, object]] = []

    for condition_key in ("wt", "mudmut", "nanobody"):
        cfg = CONDITIONS[condition_key]
        condition_records: list[LineageRecord] = []
        condition_stats = {
            "lineages_total": 0,
            "lineages_kept": 0,
            "lineages_skipped_disconnected": 0,
            "lineages_skipped_no_dpn": 0,
        }

        print("\n" + "=" * 80)
        print(f"Processing {cfg['label']} WRL files")
        print("=" * 80)

        for lobe in cfg["lobes"]:
            lobe_records, lobe_stats = analyze_lobe(
                condition_key=condition_key,
                lobe=lobe,
                min_lineage_faces=args.min_lineage_faces,
            )
            condition_records.extend(lobe_records)
            for key in condition_stats:
                condition_stats[key] += lobe_stats[key]
            print(
                f"Lobe {lobe}: kept={lobe_stats['lineages_kept']} / {lobe_stats['lineages_total']}"
                f" | skipped_disconnected={lobe_stats['lineages_skipped_disconnected']}"
                f" | skipped_no_dpn={lobe_stats['lineages_skipped_no_dpn']}"
            )

        summary = summarize_condition(condition_records, condition_stats)
        print_condition_summary(summary)

        all_records.extend(condition_records)
        all_summaries.append(summary)

    per_lineage_csv = args.outdir / "experimental_wrl_metrics_per_lineage.csv"
    summary_csv = args.outdir / "experimental_wrl_metrics_summary.csv"
    write_per_lineage_csv(per_lineage_csv, all_records)
    write_summary_csv(summary_csv, all_summaries)

    print(f"\nSaved per-lineage metrics to {per_lineage_csv}")
    print(f"Saved condition summaries to {summary_csv}")


if __name__ == "__main__":
    main()
