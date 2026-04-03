"""
visualize_lineage.py

Interactive 3D viewer for a single filtered WRL lineage.
Opens an interactive plot in the browser via plotly.

Usage
-----
# Print the index table and exit:
  python -m scripts.visualize_lineage --list

# View lineage by row number (0-based) in the index:
  python -m scripts.visualize_lineage --row 7

# View lineage by its lineage_id string:
  python -m scripts.visualize_lineage --id lin0007

Run preprocess_lineages_for_viz.py first to build the cache.

Controls (plotly)
-----------------
  Mouse drag  – rotate
  Scroll      – zoom
  Double-click a legend entry – isolate that component
  Single-click a legend entry – toggle visibility
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Colours (rgba strings for plotly)
# ---------------------------------------------------------------------------
LINEAGE_COLOR = "rgba(100, 149, 237, 0.15)"   # cornflower blue, very transparent
DPN_COLOR     = "rgba(230, 97,  0,  1.00)"    # orange, fully opaque
PROS_COLOR    = "rgba(50,  170, 90, 0.45)"    # green, semi-transparent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_index_table(rows: list[dict[str, str]]) -> None:
    has_reason = "rejection_reason" in (rows[0] if rows else {})
    header = (
        f"{'row':>5}  {'lineage_id':<12}  {'genotype':<8}  {'lobe':<8}  "
        f"{'lin_idx':>7}  {'n_dpn':>5}  {'n_pros':>6}  {'total_cells':>11}  {'volume_um3':>12}"
        + (f"  {'rejection_reason':<16}" if has_reason else "")
    )
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows):
        reason = f"  {row.get('rejection_reason', ''):<16}" if has_reason else ""
        print(
            f"{i:>5}  {row['lineage_id']:<12}  {row['genotype']:<8}  {row['lobe']:<8}  "
            f"{row['lineage_idx']:>7}  {row['n_dpn']:>5}  {row['n_pros']:>6}  "
            f"{row['total_cells']:>11}  {row['volume_um3']:>12}{reason}"
        )


def load_lineage_meshes(
    npz_path: Path,
) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    data = np.load(npz_path)

    def reconstruct(prefix: str) -> trimesh.Trimesh:
        return trimesh.Trimesh(
            vertices=data[f"{prefix}_vertices"],
            faces=data[f"{prefix}_faces"],
            process=False,
        )

    lineage_mesh = reconstruct("lin")

    dpn_meshes: list[trimesh.Trimesh] = []
    i = 0
    while f"dpn_{i}_vertices" in data:
        dpn_meshes.append(reconstruct(f"dpn_{i}"))
        i += 1

    pros_meshes: list[trimesh.Trimesh] = []
    i = 0
    while f"pros_{i}_vertices" in data:
        pros_meshes.append(reconstruct(f"pros_{i}"))
        i += 1

    return lineage_mesh, dpn_meshes, pros_meshes


def mesh_to_trace(
    mesh: trimesh.Trimesh,
    color: str,
    name: str,
    *,
    show_legend: bool = True,
    legend_group: str = "",
) -> go.Mesh3d:
    v = mesh.vertices
    f = mesh.faces
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color,
        name=name,
        legendgroup=legend_group,
        showlegend=show_legend,
        flatshading=False,
        lighting={"ambient": 0.6, "diffuse": 0.8, "specular": 0.2},
    )


def build_figure(
    lineage_mesh: trimesh.Trimesh,
    dpn_meshes: list[trimesh.Trimesh],
    pros_meshes: list[trimesh.Trimesh],
    row: dict[str, str],
) -> go.Figure:
    traces: list[go.Mesh3d] = []

    # Render order for WebGL transparency:
    #   1. Lineage hull — most transparent, background context
    #   2. Dpn — fully opaque, writes to depth buffer so it appears solid inside the cluster
    #   3. Pros — semi-transparent, rendered last so it wraps around/over the Dpn
    traces.append(
        mesh_to_trace(
            lineage_mesh,
            LINEAGE_COLOR,
            name="lineage hull",
            legend_group="lineage",
        )
    )

    for i, m in enumerate(dpn_meshes):
        traces.append(
            mesh_to_trace(
                m,
                DPN_COLOR,
                name="Dpn (neuroblast)" if i == 0 else f"Dpn {i}",
                legend_group="dpn",
                show_legend=(i == 0),
            )
        )

    for i, m in enumerate(pros_meshes):
        traces.append(
            mesh_to_trace(
                m,
                PROS_COLOR,
                name="Pros (daughter)" if i == 0 else f"Pros {i}",
                legend_group="pros",
                show_legend=(i == 0),
            )
        )

    title = (
        f"{row['lineage_id']}  |  {row['genotype']}  |  {row['lobe']}  |  "
        f"lin_idx={row['lineage_idx']}  |  "
        f"Dpn={row['n_dpn']}  Pros={row['n_pros']}  total={row['total_cells']}  "
        f"vol={row['volume_um3']} µm³"
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={"text": title, "font": {"size": 13}},
        scene={
            "xaxis_title": "x (µm)",
            "yaxis_title": "y (µm)",
            "zaxis_title": "z (µm)",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for a cached WRL lineage (opens in browser)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/exp/viz_cache"),
        help="Cache directory written by preprocess_lineages_for_viz.py (default: data/exp/viz_cache)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--list",
        action="store_true",
        help="Print the lineage index table and exit.",
    )
    group.add_argument(
        "--row",
        type=int,
        metavar="N",
        help="View lineage at row N (0-based) in the index.",
    )
    group.add_argument(
        "--id",
        type=str,
        metavar="LINEAGE_ID",
        help="View lineage by its lineage_id string (e.g. lin0007 or rej0003).",
    )
    parser.add_argument(
        "--rejected",
        action="store_true",
        help="Browse/view rejected lineages instead of kept ones.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Auto-detect rejected vs kept from ID prefix; --rejected overrides for --list/--row
    use_rejected = args.rejected or (args.id is not None and args.id.startswith("rej"))
    if use_rejected:
        index_path = args.cache_dir / "rejected_lineage_index.csv"
        mesh_dir = args.cache_dir / "rejected_meshes"
    else:
        index_path = args.cache_dir / "lineage_index.csv"
        mesh_dir = args.cache_dir / "meshes"

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run scripts/preprocess_lineages_for_viz.py first."
        )
    with index_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if args.list or (args.row is None and args.id is None):
        print_index_table(rows)
        return

    if args.row is not None:
        if args.row < 0 or args.row >= len(rows):
            raise SystemExit(f"Row {args.row} out of range (0–{len(rows)-1}).")
        row = rows[args.row]
    else:
        matches = [r for r in rows if r["lineage_id"] == args.id]
        if not matches:
            raise SystemExit(f"lineage_id '{args.id}' not found in index.")
        row = matches[0]

    npz_path = mesh_dir / f"{row['lineage_id']}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Mesh cache not found: {npz_path}")

    status = f"  [REJECTED: {row['rejection_reason']}]" if use_rejected else ""
    print(
        f"\nViewing {row['lineage_id']}  |  genotype={row['genotype']}  "
        f"lobe={row['lobe']}  lineage_idx={row['lineage_idx']}{status}"
    )
    print(
        f"  n_dpn={row['n_dpn']}  n_pros={row['n_pros']}  "
        f"total_cells={row['total_cells']}  volume={row['volume_um3']} µm³"
    )
    print("\nOpening browser…\n")

    lineage_mesh, dpn_meshes, pros_meshes = load_lineage_meshes(npz_path)
    fig = build_figure(lineage_mesh, dpn_meshes, pros_meshes, row)
    fig.show()


if __name__ == "__main__":
    main()
