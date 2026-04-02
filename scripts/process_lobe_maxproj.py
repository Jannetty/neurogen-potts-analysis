# process_lobe_maxproj.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import unary_union
from matplotlib.colors import ListedColormap
import random

from src.expdata_geo_helpers import (
    mesh_slice_polygon_2d,
    project_lineage_polygon_2d,
    choose_slice_plane_for_lineage,
    pad_grid_to_canvas,
    draw_poly_fill,
    draw_poly_outline,
)
from src.wrl_lineage_filtering import load_filtered_wrl_lobe

# ---------- Max-projection pipeline ----------


def process_lobe_maxproj(
    lineage_file,
    pros_file,
    dpn_file,
    ds=0.3,
    n_lineages=None,
    canvas_size=200,
    show=True,
    output_file=None,
    return_prevoxel_areas=False,
):
    """
    For each lineage:
      - import VRML meshes
      - assign Pros/Dpn cells to lineages (bbox-based, like process_lobe_nearest)
      - choose slice plane via PCA + Dpn logic
      - compute:
          * poly_lineage_slice : true 2D cross-section of lineage on that plane
          * poly_lineage_proj  : orthographic projection (hull) of full lineage
          * cell_maxproj_polys : max-projected Pros and Dpn polygons assigned
                                 to that lineage, intersecting the hull
      - voxelize onto a regular 2D grid with spacing ds:
          * union_mask: lineage hull ∪ Pros
          * nb_mask   : Dpn (neuroblast) mask
      - pad/center to canvas_size×canvas_size
      - collect:
          * geo_tensor[y, x, 0] = nb_mask
          * geo_tensor[y, x, 1] = union_mask
          * counts = [N_dpn, N_pros]  (for that lineage)

    If output_file is not None, saves:
        geo_tensors: (N_lineages, canvas_size, canvas_size, 2)
        counts:      (N_lineages, 2)  [N_dpn, N_pros]
        lineage_ids: (N_lineages,)
        ds:          scalar
    """

    # --- Load meshes ---
    filtered_lobe = load_filtered_wrl_lobe(lineage_file, pros_file, dpn_file)
    lineage_tm = filtered_lobe.lineage_meshes
    pros_tm = filtered_lobe.pros_meshes
    dpn_tm = filtered_lobe.dpn_meshes
    pros_to_lineage_idx = filtered_lobe.pros_to_lineage_idx
    dpn_to_lineage_idx = filtered_lobe.dpn_to_lineage_idx
    kept_lineage_indices = {item.lineage_index for item in filtered_lobe.kept_lineages}

    print("Pros per lineage (bbox assignment):")
    for i in range(len(lineage_tm)):
        print(f"  lineage {i}: {(pros_to_lineage_idx == i).sum()} pros cells")
    print("  unassigned pros:", (pros_to_lineage_idx == -1).sum())

    print("\nDpn per lineage (bbox assignment):")
    for i in range(len(lineage_tm)):
        print(f"  lineage {i}: {(dpn_to_lineage_idx == i).sum()} dpn cells")
    print("  unassigned dpn:", (dpn_to_lineage_idx == -1).sum())

    if n_lineages is None:
        n_lineages = len(lineage_tm)

    # --- Accumulators for data ---
    geo_tensors = []  # list of (H, W, 2)
    counts_list = []  # list of [N_dpn, N_pros]
    lineage_ids = []  # which lineage index each sample came from
    prevoxel_lineage_areas_um2 = (
        []
    )  # area of projected lineage hull before voxelization

    # Genotype label inferred from output_file name
    genotype_label = None
    if output_file is not None:
        name = Path(output_file).name.lower()
        if "wt" in name and "mud" not in name and "nano" not in name:
            genotype_label = "wt"
        elif "mud" in name:
            genotype_label = "mudmut"
        elif "nano" in name:
            genotype_label = "nanobody"

    labels = []  # per-lineage genotype labels (optional)

    # --- Loop over lineages ---
    for L in range(n_lineages):
        print(f"\n=== Lineage {L} ===")

        if L not in kept_lineage_indices:
            print(f"=== Lineage {L} does not pass shared WRL lineage filters; skipping ===")
            continue

        lineage_mesh_L = lineage_tm[L]

        pros_indices_L = np.where(pros_to_lineage_idx == L)[0]
        pros_meshes_L = [pros_tm[i] for i in pros_indices_L]

        dpn_indices_L = np.where(dpn_to_lineage_idx == L)[0]
        dpn_meshes_L = [dpn_tm[i] for i in dpn_indices_L]

        # --- PCA on lineage geometry ---
        all_L_vertices = lineage_mesh_L.vertices
        mean_L = all_L_vertices.mean(axis=0)
        X = all_L_vertices - mean_L
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        e1, e2, e3 = Vt
        u_axis, v_axis = e1, e2

        # --- Choose slice plane using Dpn-aware logic ---
        plane_origin, plane_normal, u_axis, v_axis = choose_slice_plane_for_lineage(
            lineage_mesh_L=lineage_mesh_L,
            dpn_meshes_L=dpn_meshes_L,
            mean_L=mean_L,
            e1=e1,
            e2=e2,
            e3=e3,
        )

        # --- True lineage slice polygon (for reference) ---
        poly_lineage_slice = mesh_slice_polygon_2d(
            lineage_mesh_L, plane_origin, plane_normal, u_axis, v_axis
        )
        if poly_lineage_slice is None:
            print(f"[Lineage {L}] No intersection with plane; skipping.")
            continue

        # --- Full lineage projection hull onto this plane ---
        buffer_radius = ds
        poly_lineage_proj = project_lineage_polygon_2d(
            lineage_mesh_L=lineage_mesh_L,
            plane_origin=plane_origin,
            u_axis=u_axis,
            v_axis=v_axis,
            buffer_radius=buffer_radius,
        )
        if poly_lineage_proj is None:
            poly_lineage_proj = poly_lineage_slice

        # --- Max-projection of all Pros + Dpn onto the same plane ---
        cell_proj_polys = []  # list of (poly, type_str)

        # Pros cells
        for mesh in pros_meshes_L:
            poly_proj = project_lineage_polygon_2d(
                lineage_mesh_L=mesh,
                plane_origin=plane_origin,
                u_axis=u_axis,
                v_axis=v_axis,
                buffer_radius=0.0,
            )
            if poly_proj is None or poly_proj.is_empty:
                continue
            # only keep Pros that intersect the lineage hull projection
            if not poly_proj.intersects(poly_lineage_proj):
                continue
            cell_proj_polys.append((poly_proj, "pros"))

        # Dpn cells
        for mesh in dpn_meshes_L:
            poly_proj = project_lineage_polygon_2d(
                lineage_mesh_L=mesh,
                plane_origin=plane_origin,
                u_axis=u_axis,
                v_axis=v_axis,
                buffer_radius=0.0,
            )
            if poly_proj is None or poly_proj.is_empty:
                continue
            if not poly_proj.intersects(poly_lineage_proj):
                continue
            cell_proj_polys.append((poly_proj, "dpn"))

        if not cell_proj_polys:
            print(f"[Lineage {L}] No max-projected cell polygons; skipping.")
            continue

        # --- Build union polygons for voxelization ---
        pros_polys = [poly for poly, ctype in cell_proj_polys if ctype == "pros"]
        dpn_polys = [poly for poly, ctype in cell_proj_polys if ctype == "dpn"]

        # union of lineage hull + all Pros = "lineage+Pros" mask
        if pros_polys:
            union_poly = unary_union([poly_lineage_proj] + pros_polys)
        else:
            union_poly = poly_lineage_proj

        # union of all Dpn = "neuroblast" mask
        nb_poly = unary_union(dpn_polys) if dpn_polys else None

        # area in um^2 before voxelization (projection hull on slice plane)
        prevoxel_lineage_areas_um2.append(float(poly_lineage_proj.area))

        # --- Voxelize onto a regular grid with spacing ds ---
        xmin, ymin, xmax, ymax = poly_lineage_proj.bounds
        pad = ds * 0.5
        xmin -= pad
        ymin -= pad
        xmax += pad
        ymax += pad

        width = xmax - xmin
        height = ymax - ymin

        nx = int(np.ceil(width / ds))
        ny = int(np.ceil(height / ds))

        max_voxels = 1_000_000
        total_voxels = nx * ny
        if total_voxels > max_voxels:
            print(
                f"[Lineage {L}] grid {ny}x{nx} = {total_voxels} voxels "
                f"exceeds max_voxels={max_voxels}; increase ds."
            )
            continue

        union_mask = np.zeros((ny, nx), dtype=bool)
        nb_mask = np.zeros((ny, nx), dtype=bool)

        voxel_area = ds * ds
        min_fraction = 0.5

        for iy in range(ny):
            y0 = ymin + iy * ds
            y1 = y0 + ds
            for ix in range(nx):
                x0 = xmin + ix * ds
                x1 = x0 + ds
                voxel = box(x0, y0, x1, y1)

                # union of lineage hull + Pros
                if union_poly is not None:
                    inter_area = voxel.intersection(union_poly).area
                    if inter_area / voxel_area >= min_fraction:
                        union_mask[iy, ix] = True

                # neuroblast (Dpn) mask
                if nb_poly is not None:
                    inter_area_nb = voxel.intersection(nb_poly).area
                    if inter_area_nb / voxel_area >= min_fraction:
                        nb_mask[iy, ix] = True

        # --- Convert masks to centered canvas tensors ---
        # use pad_grid_to_canvas to build a canvas_size×canvas_size grid
        extent = (xmin, xmax, ymin, ymax)

        union_canvas, canvas_extent = pad_grid_to_canvas(
            union_mask.astype(np.int32),
            extent,
            ds,
            canvas_size=canvas_size,
        )
        nb_canvas, _ = pad_grid_to_canvas(
            nb_mask.astype(np.int32),
            extent,
            ds,
            canvas_size=canvas_size,
        )

        # geometry tensor: (H, W, 2), channel 0 = nb, channel 1 = union
        geo_tensor = np.stack(
            [nb_canvas, union_canvas],
            axis=-1,
        ).astype(np.float32)

        # counts: how many Dpn and Pros cells participate in this lineage slice
        N_dpn = len(dpn_polys)
        N_pros = len(pros_polys)

        # per-NB individual voxel counts (not union) for correct avg NB area
        dpn_sum_voxels = 0
        for dpn_poly in dpn_polys:
            for iy in range(ny):
                y0 = ymin + iy * ds
                y1 = y0 + ds
                for ix in range(nx):
                    x0 = xmin + ix * ds
                    x1 = x0 + ds
                    voxel = box(x0, y0, x1, y1)
                    if voxel.intersection(dpn_poly).area / voxel_area >= min_fraction:
                        dpn_sum_voxels += 1

        geo_tensors.append(geo_tensor)
        counts_list.append([N_dpn, N_pros, dpn_sum_voxels])
        lineage_ids.append(L)

        if genotype_label is not None:
            labels.append(genotype_label)

        # --- Visualization: hull + max-projected cells ---
        if show:
            fig, ax = plt.subplots(figsize=(6, 6))

            # 1) full lineage projection hull as light gray background
            draw_poly_fill(
                ax,
                poly_lineage_proj,
                color="lightgray",
                alpha=0.3,
                zorder=0,
            )

            # 2) true slice outline for reference (dashed)
            if poly_lineage_slice is not None:
                draw_poly_outline(
                    ax,
                    poly_lineage_slice,
                    color="black",
                    linewidth=1.0,
                    linestyle="--",
                    zorder=1,
                )

            # 3) max-projected cells
            pros_color = "#55a2a9"
            dpn_color = "#6b4ea3"

            for poly, ctype in cell_proj_polys:
                col = pros_color if ctype == "pros" else dpn_color
                draw_poly_fill(
                    ax,
                    poly,
                    color=col,
                    alpha=0.8 if ctype == "dpn" else 0.6,
                    zorder=2 if ctype == "pros" else 3,
                )

            # 4) axis settings: zoom to lineage projection bounds
            xmin_plot, ymin_plot, xmax_plot, ymax_plot = poly_lineage_proj.bounds
            dx = xmax_plot - xmin_plot
            dy = ymax_plot - ymin_plot
            pad_plot = 0.05 * max(dx, dy)
            ax.set_xlim(xmin_plot - pad_plot, xmax_plot + pad_plot)
            ax.set_ylim(ymin_plot - pad_plot, ymax_plot + pad_plot)

            ax.set_aspect("equal", "box")
            ax.set_xlabel("PC1 coordinate (µm)")
            ax.set_ylabel("PC2 coordinate (µm)")
            ax.set_title(f"Lineage {L}: max-projected Pros/Dpn onto slice plane")

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color=pros_color,
                    linestyle="",
                    label="Pros (max proj)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color=dpn_color,
                    linestyle="",
                    label="Deadpan (max proj)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="lightgray",
                    linestyle="",
                    label="Lineage hull (projection)",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    label="Lineage slice (true)",
                ),
            ]
            ax.legend(handles=handles, loc="upper right")
            plt.tight_layout()
            plt.show()

            # --- Visualization: voxelized union + Dpn masks ---
            fig2, ax2 = plt.subplots(figsize=(6, 6))

            grid = np.zeros((ny, nx), dtype=int)
            grid[union_mask] = 1
            grid[nb_mask] = 2  # Dpn overwrites union where both are True

            cmap = ListedColormap(["none", "#55a2a9", "#6b4ea3"])
            grid_masked = np.ma.masked_where(grid == 0, grid)

            ax2.imshow(
                grid_masked,
                origin="lower",
                extent=(xmin, xmax, ymin, ymax),
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=2,
            )

            ax2.set_aspect("equal", "box")
            ax2.set_xlabel("PC1 coordinate (µm)")
            ax2.set_ylabel("PC2 coordinate (µm)")
            ax2.set_title(f"Lineage {L}: voxelized union(mask) + Dpn(mask)")

            handles2 = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="#55a2a9",
                    linestyle="",
                    label="Lineage ∪ Pros voxels",
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="#6b4ea3",
                    linestyle="",
                    label="Dpn voxels",
                ),
            ]
            ax2.legend(handles=handles2, loc="upper right")
            plt.tight_layout()
            plt.show()

    # --- Save data if requested ---
    if not geo_tensors:
        print("No valid lineages; nothing to save.")
        if return_prevoxel_areas:
            return None, None, None, None
        return None, None, None

    geo_arr = np.stack(geo_tensors, axis=0)  # (N, H, W, 2)
    counts_arr = np.asarray(counts_list, dtype=np.float32)  # (N, 2)
    lineage_ids_arr = np.asarray(lineage_ids, dtype=int)

    # Turn labels into an array if we collected any
    labels_arr = None
    if labels:
        labels_arr = np.array(labels, dtype="U16")

    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = dict(
            geo=geo_arr,
            counts=counts_arr,
            lineage_ids=lineage_ids_arr,
            ds=ds,
        )
        if labels_arr is not None:
            save_kwargs["labels"] = labels_arr

        np.savez_compressed(output_file, **save_kwargs)
        print(f"\nSaved tensors → {output_file}")
        if labels_arr is not None:
            print(f"  labels shape: {labels_arr.shape}, first few: {labels_arr[:5]}")

    if prevoxel_lineage_areas_um2:
        area_arr = np.asarray(prevoxel_lineage_areas_um2, dtype=float)
        print(
            "Pre-voxel lineage projection area summary "
            f"(um^2): mean={area_arr.mean():.3f} std={area_arr.std():.3f} N={area_arr.size:d}"
        )
    else:
        area_arr = np.array([], dtype=float)
        print("Pre-voxel lineage projection area summary (um^2): no data")

    if return_prevoxel_areas:
        return geo_arr, counts_arr, lineage_ids_arr, area_arr
    return geo_arr, counts_arr, lineage_ids_arr


# --------------- Sanity check that imports pipeline-created files -----------
def show_sample(geo, counts, sample_idx, title_prefix=""):
    """
    geo: (N, H, W, 2)
    counts: (N, 2)
    sample_idx: int
    """

    img = geo[sample_idx]
    N_dpn, N_pros = counts[sample_idx, 0], counts[sample_idx, 1]

    # --- Print scalar counts and voxel occupancy ---
    nb_voxels = int(img[..., 0].sum())
    union_voxels = int(img[..., 1].sum())

    print(f"{title_prefix} sample idx={sample_idx}")
    print(f"  scalar counts:   N_dpn = {int(N_dpn)}, N_pros = {int(N_pros)}")
    print(
        f"  voxel occupancy: nb_mask voxels = {nb_voxels}, union_mask voxels = {union_voxels}"
    )

    # --- Plots ---
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img[..., 0], cmap="Purples")
    axs[0].set_title(f"{title_prefix} NB mask\n(count={int(N_dpn)})")
    axs[0].axis("off")

    axs[1].imshow(img[..., 1], cmap="Greys")
    axs[1].set_title(f"{title_prefix} Union mask\n(count={int(N_pros)})")
    axs[1].axis("off")

    # Overlay NB on union for sanity
    overlay = img[..., 1].copy()
    overlay[img[..., 0] > 0] = 2  # NB overwrites union
    axs[2].imshow(overlay, cmap="viridis")
    axs[2].set_title("Overlay")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def check_values(name, geo):
    print(f"\nValue check for {name}:")
    print("  min:", geo.min(), "max:", geo.max())
    unique_vals = np.unique(geo)
    print(f"  unique values (first 10): {unique_vals[:10]}")
    if not np.all((geo == 0) | (geo == 1)):
        print("⚠️  WARNING: non-binary values detected")


def random_sample_idx(geo):
    return random.randint(0, geo.shape[0] - 1)


def sanity_check(file_wt, file_mud, file_nano):
    def load_and_report(path, name):
        print(f"\nLoading {name} file:", path)
        data = np.load(path)

        geo = data["geo"]  # (N, H, W, 2)
        counts = data["counts"]  # (N, 2)
        lineage_ids = data["lineage_ids"]
        ds = data["ds"]
        labels = data.get("labels", None)

        print(f"{name} shapes:")
        print("  geo:", geo.shape)
        print("  counts:", counts.shape)
        print("  lineage_ids:", lineage_ids.shape)
        print("  ds:", ds)
        if labels is not None:
            print(f"  labels: {labels.shape}, unique:", np.unique(labels))

        print(f"\n{name} count summary:")
        print("  N_dpn  min/max:", counts[:, 0].min(), counts[:, 0].max())
        print("  N_pros min/max:", counts[:, 1].min(), counts[:, 1].max())

        check_values(name, geo)

        print(f"\nPlotting sample {name} lineage...")
        idx = random_sample_idx(geo)
        show_sample(geo, counts, idx, name)

    # Run for each genotype
    load_and_report(file_wt, "WT")
    load_and_report(file_mud, "Mudmut")
    load_and_report(file_nano, "Nanobody")

    print("\nSanity check complete.\n")


def compare_npz_files(file1, file2, atol=0.0):
    """
    Compare two NPZ files produced by process_lobe_maxproj.

    Prints:
      - shape equality for geo / counts / lineage_ids
      - whether arrays are exactly equal (or within atol)
      - max absolute differences if not exactly equal
    """
    f1 = np.load(file1)
    f2 = np.load(file2)

    print(f"\n=== Comparing NPZ files ===")
    print(f"  file1: {file1}")
    print(f"  file2: {file2}")

    keys = ["geo", "counts", "lineage_ids", "ds"]
    for k in keys:
        if k not in f1 or k not in f2:
            print(f"  key '{k}' missing in one of the files")
            continue

        a = f1[k]
        b = f2[k]

        # ds is scalar; handle separately
        if k == "ds":
            print(f"\nKey: ds")
            print(f"  file1 ds: {a}")
            print(f"  file2 ds: {b}")
            if np.allclose(a, b, atol=atol):
                print("  -> ds values match (within atol).")
            else:
                print("  -> ds values differ.")
            continue

        print(f"\nKey: {k}")
        print(f"  file1 shape: {a.shape}")
        print(f"  file2 shape: {b.shape}")

        if a.shape != b.shape:
            print("  -> shapes differ (files are NOT the same).")
            continue

        equal = np.array_equal(a, b)
        close = np.allclose(a, b, atol=atol)

        if equal:
            print("  -> arrays are EXACTLY equal.")
        elif close:
            max_diff = np.max(np.abs(a - b))
            print("  -> arrays are equal within atol.")
            print(f"     max |diff| = {max_diff}")
        else:
            max_diff = np.max(np.abs(a - b))
            print("  -> arrays differ beyond atol.")
            print(f"     max |diff| = {max_diff}")


def run_all_wt_lobes(
    lobes=(1, 2, 7, 8, 13, 15),
    ds=0.3,
    canvas_size=200,
):
    """
    Run process_lobe_maxproj on all WT (Control) WRL triplets and save NPZ files
    to data/exp_wt_npz/.

    Each lobe is expected to have files:
      data/exp/wrl_files/Control/lobe{lobe}_control_lineages.wrl
      data/exp/wrl_files/Control/lobe{lobe}_control_pros.wrl
      data/exp/wrl_files/Control/lobe{lobe}_control_dpn.wrl
    """

    base_dir = Path("data/exp/wrl_files/Control")
    out_dir = Path("data/exp/npz_files/exp_wt_npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    group_area_values = []

    for lobe in lobes:
        print("\n" + "=" * 80)
        print(f"Processing WT lobe {lobe}")
        print("=" * 80)

        lineage_file = base_dir / f"lobe{lobe}_control_lineages.wrl"
        pros_file = base_dir / f"lobe{lobe}_control_pros.wrl"
        dpn_file = base_dir / f"lobe{lobe}_control_dpn.wrl"

        # Match your existing naming convention:
        output_file = out_dir / f"lobe{lobe}_wt_maxproj.npz"

        print(f"  lineage_file: {lineage_file}")
        print(f"  pros_file:    {pros_file}")
        print(f"  dpn_file:     {dpn_file}")
        print(f"  output_file:  {output_file}")

        geo, counts, lineage_ids, area_arr = process_lobe_maxproj(
            lineage_file=lineage_file,
            pros_file=pros_file,
            dpn_file=dpn_file,
            ds=ds,
            n_lineages=None,
            canvas_size=canvas_size,
            show=False,  # visualize each lineage
            output_file=output_file,
            return_prevoxel_areas=True,
        )

        if geo is None:
            print(f"⚠️  No valid lineages for WT lobe {lobe}; nothing saved.")
        else:
            if area_arr is not None and area_arr.size > 0:
                group_area_values.extend(area_arr.tolist())
            print(
                f"✅ Finished WT lobe {lobe}: "
                f"{geo.shape[0]} lineages → {output_file}"
            )

    if group_area_values:
        group_area_arr = np.asarray(group_area_values, dtype=float)
        print(
            "\nWT pre-voxel lineage projection area summary "
            f"(um^2): mean={group_area_arr.mean():.3f} std={group_area_arr.std():.3f} "
            f"N={group_area_arr.size:d}"
        )
    else:
        print("\nWT pre-voxel lineage projection area summary (um^2): no data")


# ------ Functions for running all experimental data through pipeline ----
def run_all_mud_lobes(
    lobes=(110, 113, 116, 119),
    ds=0.3,
    canvas_size=200,
):
    """
    Run process_lobe_maxproj on all Mud WRL triplets and save NPZ files
    to data/exp_mud_npz/.

    Each lobe is expected to have files:
      data/exp/wrl_files/Mud/lobe{lobe}_mud_lineages.wrl
      data/exp/wrl_files/Mud/lobe{lobe}_mud_pros.wrl
      data/exp/wrl_files/Mud/lobe{lobe}_mud_dpn.wrl
    """

    base_dir = Path("data/exp/wrl_files/Mud")
    out_dir = Path("data/exp/npz_files/exp_mud_npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    group_area_values = []

    for lobe in lobes:
        print("\n" + "=" * 80)
        print(f"Processing Mud lobe {lobe}")
        print("=" * 80)

        lineage_file = base_dir / f"lobe{lobe}_mud_lineages.wrl"
        pros_file = base_dir / f"lobe{lobe}_mud_pros.wrl"
        dpn_file = base_dir / f"lobe{lobe}_mud_dpn.wrl"

        output_file = out_dir / f"lobe{lobe}_mudmut_maxproj.npz"

        print(f"  lineage_file: {lineage_file}")
        print(f"  pros_file:    {pros_file}")
        print(f"  dpn_file:     {dpn_file}")
        print(f"  output_file:  {output_file}")

        geo, counts, lineage_ids, area_arr = process_lobe_maxproj(
            lineage_file=lineage_file,
            pros_file=pros_file,
            dpn_file=dpn_file,
            ds=ds,
            n_lineages=None,
            canvas_size=canvas_size,
            show=False,  # visualize each lineage
            output_file=output_file,
            return_prevoxel_areas=True,
        )

        if geo is None:
            print(f"⚠️  No valid lineages for lobe {lobe}; nothing saved.")
        else:
            if area_arr is not None and area_arr.size > 0:
                group_area_values.extend(area_arr.tolist())
            print(
                f"✅ Finished lobe {lobe}: " f"{geo.shape[0]} lineages → {output_file}"
            )

    if group_area_values:
        group_area_arr = np.asarray(group_area_values, dtype=float)
        print(
            "\nMUD pre-voxel lineage projection area summary "
            f"(um^2): mean={group_area_arr.mean():.3f} std={group_area_arr.std():.3f} "
            f"N={group_area_arr.size:d}"
        )
    else:
        print("\nMUD pre-voxel lineage projection area summary (um^2): no data")


def run_all_nanobody_lobes(
    lobes=(16, 17, 19, 20, 21, 22),
    ds=0.3,
    canvas_size=200,
):
    """
    Run process_lobe_maxproj on all Nanobody WRL triplets and save NPZ files
    to data/exp_nanobody_npz/.

    Each lobe is expected to have files:
      data/exp/wrl_files/Nanobody/lobe{lobe}_Nanobody_lineages.wrl
      data/exp/wrl_files/Nanobody/lobe{lobe}_Nanobody_pros.wrl
      data/exp/wrl_files/Nanobody/lobe{lobe}_Nanobody_dpn.wrl
    """

    base_dir = Path("data/exp/wrl_files/Nanobody")
    out_dir = Path("data/exp/npz_files/exp_nanobody_npz")
    out_dir.mkdir(parents=True, exist_ok=True)
    group_area_values = []

    for lobe in lobes:
        print("\n" + "=" * 80)
        print(f"Processing Nanobody lobe {lobe}")
        print("=" * 80)

        lineage_file = base_dir / f"lobe{lobe}_Nanobody_lineages.wrl"
        pros_file = base_dir / f"lobe{lobe}_Nanobody_pros.wrl"
        dpn_file = base_dir / f"lobe{lobe}_Nanobody_dpn.wrl"

        output_file = out_dir / f"lobe{lobe}_nanobody_maxproj.npz"

        print(f"  lineage_file: {lineage_file}")
        print(f"  pros_file:    {pros_file}")
        print(f"  dpn_file:     {dpn_file}")
        print(f"  output_file:  {output_file}")

        geo, counts, lineage_ids, area_arr = process_lobe_maxproj(
            lineage_file=lineage_file,
            pros_file=pros_file,
            dpn_file=dpn_file,
            ds=ds,
            n_lineages=None,
            canvas_size=canvas_size,
            show=False,  # visualize each lineage
            output_file=output_file,
            return_prevoxel_areas=True,
        )

        if geo is None:
            print(f"⚠️  No valid lineages for Nanobody lobe {lobe}; nothing saved.")
        else:
            if area_arr is not None and area_arr.size > 0:
                group_area_values.extend(area_arr.tolist())
            print(
                f"✅ Finished Nanobody lobe {lobe}: "
                f"{geo.shape[0]} lineages → {output_file}"
            )

    if group_area_values:
        group_area_arr = np.asarray(group_area_values, dtype=float)
        print(
            "\nNANOBODY pre-voxel lineage projection area summary "
            f"(um^2): mean={group_area_arr.mean():.3f} std={group_area_arr.std():.3f} "
            f"N={group_area_arr.size:d}"
        )
    else:
        print("\nNANOBODY pre-voxel lineage projection area summary (um^2): no data")


if __name__ == "__main__":
    run_all_wt_lobes()
    run_all_mud_lobes()
    run_all_nanobody_lobes()

    # Sanity check one representative NPZ from each genotype.
    wt_file = Path("data/exp/npz_files/exp_wt_npz/lobe1_wt_maxproj.npz")
    mud_file = Path("data/exp/npz_files/exp_mud_npz/lobe110_mudmut_maxproj.npz")
    nano_file = Path("data/exp/npz_files/exp_nanobody_npz/lobe16_nanobody_maxproj.npz")

    sanity_check(wt_file, mud_file, nano_file)
