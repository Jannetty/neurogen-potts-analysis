# compare_wrl_wt_vs_mud_lineage_volume_and_dpn_counts.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from scipy.stats import ttest_ind

from src.wrl_lineage_filtering import WrlLineageFilterConfig, load_filtered_wrl_lobe


# ---------------------------- Data containers ----------------------------

@dataclass(frozen=True)
class LineageRecord:
    genotype: str
    lobe: str
    lineage_idx: int
    volume_um3: float
    ndpn_cells: int
    npros_cells: int
    n_cells_total: int


# ---------------------------- File discovery ----------------------------

def find_lobe_triplets(base_dir: Path, genotype: str) -> List[Tuple[str, Path, Path, Path]]:
    """
    Returns list of (lobe_name, lineage_file, dpn_file).

    Assumes file naming:
      Control:  lobe{X}_control_lineages.wrl, lobe{X}_control_dpn.wrl
      Mud:      lobe{X}_mud_lineages.wrl,     lobe{X}_mud_dpn.wrl
    """
    if genotype == "wt":
        sub = base_dir / "Control"
        lineage_glob = "*_control_lineages.wrl"
        dpn_suffix = "_control_dpn.wrl"
        pros_suffix = "_control_pros.wrl"
        lineage_suffix = "_control_lineages.wrl"
    elif genotype == "mudmut":
        sub = base_dir / "Mud"
        lineage_glob = "*_mud_lineages.wrl"
        dpn_suffix = "_mud_dpn.wrl"
        pros_suffix = "_mud_pros.wrl"
        lineage_suffix = "_mud_lineages.wrl"
    else:
        raise ValueError(f"Unsupported genotype: {genotype}")

    if not sub.exists():
        raise FileNotFoundError(f"Missing directory: {sub}")

    lineage_files = sorted(sub.glob(lineage_glob))
    if not lineage_files:
        raise FileNotFoundError(f"No lineage files found in {sub} matching {lineage_glob}")

    triplets: List[Tuple[str, Path, Path, Path]] = []
    for lf in lineage_files:
        name = lf.name
        if not name.endswith(lineage_suffix):
            continue
        lobe_name = name.replace(lineage_suffix, "")  # e.g. "lobe1"
        dpn_file = sub / f"{lobe_name}{dpn_suffix}"
        if not dpn_file.exists():
            raise FileNotFoundError(f"Expected dpn file not found: {dpn_file}")
        pros_file = sub / f"{lobe_name}{pros_suffix}"
        if not pros_file.exists():
            raise FileNotFoundError(f"Expected pros file not found: {pros_file}")

        triplets.append((lobe_name, lf, dpn_file, pros_file))

    return triplets


# ---------------------------- Geometry utilities ----------------------------

def mesh_volume_um3(mesh: trimesh.Trimesh) -> float:
    """
    Return mesh volume in µm³.
    Assumes WRL coordinates are already in microns.
    """
    vol = float(mesh.volume) / 1000
    if not np.isfinite(vol):
        raise ValueError("Mesh volume is not finite.")
    # Some meshes may have flipped normals → negative volume
    return abs(vol)


def compute_lineage_volumes_and_dpn_counts(
    lineage_file: Path,
    dpn_file: Path,
    pros_file: Path,
    genotype: str,
    lobe_name: str,
    min_lineage_faces: int = 50,
    dpn_min_fraction: float = 0.60,
    pros_min_fraction: float = 0.95,
) -> List[LineageRecord]:
    """
    - Loads lineage meshes and Dpn meshes.
    - Computes per-lineage volume (µm^3) from lineage mesh.
    - Assigns Dpn cells to lineage indices using bbox overlap (strict).
    - Skips disconnected lineages using the same min-face rule as
      scripts/analyze_experimental_wrl_metrics.py.
    - Skips lineages with 0 neuroblasts (Dpn cells).
    - Returns one record per lineage.
    """
    filtered_lobe = load_filtered_wrl_lobe(
        lineage_file,
        pros_file,
        dpn_file,
        config=WrlLineageFilterConfig(
            min_lineage_faces=min_lineage_faces,
            min_pros_fraction=pros_min_fraction,
            min_dpn_fraction=dpn_min_fraction,
        ),
    )
    dpn_meshes = filtered_lobe.dpn_meshes

    if len(filtered_lobe.lineage_meshes) == 0:
        return []

    records: List[LineageRecord] = []
    for filtered_lineage in filtered_lobe.kept_lineages:
        L = filtered_lineage.lineage_index
        lin_mesh = filtered_lineage.lineage_mesh
        ndpn = filtered_lineage.n_dpn
        npros = filtered_lineage.n_pros
        n_total = ndpn + npros

        vol_um3 = mesh_volume_um3(lin_mesh)

        records.append(
            LineageRecord(
                genotype=genotype,
                lobe=lobe_name,
                lineage_idx=L,
                volume_um3=float(vol_um3),
                ndpn_cells=ndpn,
                npros_cells=npros,
                n_cells_total=n_total,
            )
        )

    return records


# ---------------------------- Plotting ----------------------------

def welch_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return np.nan
    return float(ttest_ind(a, b, equal_var=False).pvalue)


def plot_histogram_compare(
    wt_vals: np.ndarray,
    mud_vals: np.ndarray,
    xlabel: str,
    title: str,
    outpath: Path,
    bins: str | int = "auto",
    show: bool = False,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # ---- Font sizes ----
    TITLE_FS = 18
    LABEL_FS = 16
    TICK_FS = 13
    LEGEND_FS = 13

    # ---- Styling ----
    WT_COLOR = "tab:blue"
    MUD_COLOR = "tab:orange"
    BAR_ALPHA = 0.35
    EDGE_ALPHA = 0.25
    EDGE_LW = 0.8

    MEAN_LW = 3.0
    MEAN_ALPHA = 1.0
    MEAN_LS = "-"

    # Shared bins
    if len(wt_vals) and len(mud_vals):
        all_vals = np.concatenate([wt_vals, mud_vals])
    elif len(wt_vals):
        all_vals = wt_vals
    elif len(mud_vals):
        all_vals = mud_vals
    else:
        raise ValueError("Both wt_vals and mud_vals are empty.")

    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    ax.hist(
        wt_vals,
        bins=bin_edges,
        alpha=BAR_ALPHA,
        label=f"WT (N = {len(wt_vals)})",
        density=False,
        edgecolor=(0, 0, 0, EDGE_ALPHA),
        linewidth=EDGE_LW,
    )
    ax.hist(
        mud_vals,
        bins=bin_edges,
        alpha=BAR_ALPHA,
        label=f"Mudmut (N = {len(mud_vals)})",
        density=False,
        edgecolor=(0, 0, 0, EDGE_ALPHA),
        linewidth=EDGE_LW,
    )

    ax.set_title(title, fontsize=TITLE_FS, wrap=True)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel("Count", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    # Mean lines
    wt_mean = float(np.mean(wt_vals)) if len(wt_vals) else np.nan
    mud_mean = float(np.mean(mud_vals)) if len(mud_vals) else np.nan

    if np.isfinite(wt_mean):
        ax.axvline(
            wt_mean,
            color=WT_COLOR,
            linestyle=MEAN_LS,
            linewidth=MEAN_LW,
            alpha=MEAN_ALPHA,
            label=f"WT mean = {wt_mean:.2g}",
            zorder=5,
        )
    if np.isfinite(mud_mean):
        ax.axvline(
            mud_mean,
            color=MUD_COLOR,
            linestyle=MEAN_LS,
            linewidth=MEAN_LW,
            alpha=MEAN_ALPHA,
            label=f"mudmut mean = {mud_mean:.2g}",
            zorder=5,
        )

    # Welch's t-test p-value
    p = welch_pvalue(wt_vals, mud_vals)
    if np.isfinite(p):
        ax.plot([], [], " ", label=f"Welch's t-test: p = {p:.2e}")

    ax.legend(loc="best", fontsize=LEGEND_FS, frameon=False)

    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    fig.savefig(outpath, dpi=250)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------- Main ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-lineage 3D volumes directly from WRL lineage meshes and "
            "per-lineage Dpn neuroblast counts via bbox assignment, then plot WT vs mudmut histograms."
        )
    )
    parser.add_argument(
        "--base_dir",
        type=Path,
        default=Path("data/exp/wrl_files"),
        help="Base directory containing Control/ and Mud/ (default: data/exp/wrl_files)",
    )
    parser.add_argument(
        "--min_lineage_faces",
        type=int,
        default=50,
        help="Minimum face count for connected lineage components (default: 50)",
    )
    parser.add_argument(
        "--dpn_min_fraction",
        type=float,
        default=0.60,
        help="Min bbox overlap fraction for assigning Dpn cells to lineages (default: 0.60)",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/exp/plots"),
        help="Directory to save plots (default: data/exp/plots)",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="auto",
        help="Histogram bins (int or numpy rule like 'auto', 'fd', 'doane') (default: auto)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively (also saves).",
    )
    parser.add_argument(
        "--single-dpn-only",
        action="store_true",
        help="Restrict both WT and mudmut lineages to those with exactly one Dpn+ cell.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )

    # bins can be int or string rule
    try:
        bins: str | int = int(args.bins)
    except ValueError:
        bins = args.bins

    wt_triplets = find_lobe_triplets(args.base_dir, genotype="wt")
    mud_triplets = find_lobe_triplets(args.base_dir, genotype="mudmut")

    all_records: List[LineageRecord] = []

    # WT
    for lobe_name, lineage_file, dpn_file, pros_file in wt_triplets:
        print(f"[WT]  {lobe_name}: {lineage_file.name} + {dpn_file.name}")
        recs = compute_lineage_volumes_and_dpn_counts(
            lineage_file=lineage_file,
            dpn_file=dpn_file,
            pros_file=pros_file,
            genotype="wt",
            lobe_name=lobe_name,
            min_lineage_faces=args.min_lineage_faces,
            dpn_min_fraction=args.dpn_min_fraction,
            pros_min_fraction=0.95,
        )
        all_records.extend(recs)

    # Mud
    for lobe_name, lineage_file, dpn_file, pros_file in mud_triplets:
        print(f"[Mud] {lobe_name}: {lineage_file.name} + {dpn_file.name}")
        recs = compute_lineage_volumes_and_dpn_counts(
            lineage_file=lineage_file,
            dpn_file=dpn_file,
            pros_file=pros_file,
            genotype="mudmut",
            lobe_name=lobe_name,
            min_lineage_faces=args.min_lineage_faces,
            dpn_min_fraction=args.dpn_min_fraction,
            pros_min_fraction=0.95,
        )
        all_records.extend(recs)

    if not all_records:
        raise RuntimeError("No lineage records produced. Check that WRL files load correctly.")

    if args.single_dpn_only:
        before_total = len(all_records)
        all_records = [r for r in all_records if r.ndpn_cells == 1]
        print(
            "\nFiltered to single-Dpn lineages across both genotypes:"
            f" kept {len(all_records)}/{before_total} lineages (ndpn_cells == 1)"
        )

    # Split by genotype
    wt_vol = np.array([r.volume_um3 for r in all_records if r.genotype == "wt"], dtype=float)
    mud_vol = np.array([r.volume_um3 for r in all_records if r.genotype == "mudmut"], dtype=float)

    wt_ndpn = np.array([r.ndpn_cells for r in all_records if r.genotype == "wt"], dtype=float)
    mud_ndpn = np.array([r.ndpn_cells for r in all_records if r.genotype == "mudmut"], dtype=float)

    wt_ncells = np.array([r.n_cells_total for r in all_records if r.genotype == "wt"], dtype=float)
    mud_ncells = np.array([r.n_cells_total for r in all_records if r.genotype == "mudmut"], dtype=float)

    wt_npros = np.array([r.npros_cells for r in all_records if r.genotype == "wt"], dtype=float)
    mud_npros = np.array([r.npros_cells for r in all_records if r.genotype == "mudmut"], dtype=float)

    print("\nSummary:")
    print(f"  WT lineages:    {len(wt_vol)}")
    print(f"  mudmut lineages:{len(mud_vol)}")
    print(f"  min_lineage_faces: {args.min_lineage_faces}")
    print(f"  dpn_min_fraction: {args.dpn_min_fraction}")
    print(f"  single_dpn_only: {args.single_dpn_only}")

    filename_suffix = "_singleDpnOnly" if args.single_dpn_only else ""

    # Plot 1: lineage volumes
    plot_histogram_compare(
        wt_vals=wt_vol,
        mud_vals=mud_vol,
        xlabel="Lineage volume (µm³)",
        title="Lineage volumes",
        outpath=args.out_dir / f"hist_lineage_volume_wt_vs_mud{filename_suffix}.png",
        bins=bins,
        show=args.show,
    )

    # Plot 2: dpn counts
    plot_histogram_compare(
        wt_vals=wt_ndpn,
        mud_vals=mud_ndpn,
        xlabel="Number of neuroblasts per lineage",
        title="Neuroblast counts per lineage",
        outpath=args.out_dir / f"hist_dpn_counts_wt_vs_mud{filename_suffix}.png",
        bins=bins,
        show=args.show,
    )

    # Plot 3: cell counts
    plot_histogram_compare(
        wt_vals=wt_ncells,
        mud_vals=mud_ncells,
        xlabel="Number of labeled cells per lineage (Dpn + Pros)",
        title="Distribution of labeled cell counts per lineage (WT vs mudmut)",
        outpath=args.out_dir / f"hist_total_labeled_cells_wt_vs_mud{filename_suffix}.png",
        bins=bins,
        show=args.show,
    )

    # Plot 4: non-neuroblast cell counts (Pros only)
    plot_histogram_compare(
        wt_vals=wt_npros,
        mud_vals=mud_npros,
        xlabel="Non-neuroblast cells per lineage (Pros)",
        title="Non-neuroblast (Pros+) counts per lineage",
        outpath=args.out_dir / f"hist_pros_counts_wt_vs_mud{filename_suffix}.png",
        bins=bins,
        show=args.show,
    )
    print(f"\nSaved plots to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
