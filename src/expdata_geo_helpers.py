# expdata_geo_helpers.py
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import trimesh
from shapely.geometry import Polygon, box, MultiPolygon, MultiPoint, Point
from shapely.ops import unary_union
from matplotlib.colors import ListedColormap
from shapely.strtree import STRtree
from scipy.spatial import cKDTree

# ---------- VRML parsing helpers ----------


def _parse_vertices(pts_block):
    float_strings = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", pts_block)
    floats = np.array(float_strings, dtype=float)
    if len(floats) % 3 != 0:
        floats = floats[: len(floats) // 3 * 3]
    return floats.reshape(-1, 3)


def _parse_faces(idx_block):
    idx_strings = re.findall(r"-?\d+", idx_block)
    indices = [int(s) for s in idx_strings]
    faces = []
    current = []
    for v in indices:
        if v == -1:
            if len(current) >= 3:
                # fan triangulation
                for k in range(1, len(current) - 1):
                    faces.append([current[0], current[k], current[k + 1]])
            current = []
        else:
            current.append(v)
    return np.array(faces, dtype=int) if faces else None


def load_vrml_meshes(vrml_path):
    """
    Parse a VRML2 file and return a list of (vertices, faces) meshes.
    """
    text = Path(vrml_path).read_text(errors="ignore")

    coord_pattern = re.compile(r"Coordinate\s*{[^}]*?point\s*\[([^]]*)\]", re.S)
    index_pattern = re.compile(r"coordIndex\s*\[([^]]*)\]", re.S)

    coord_matches = list(coord_pattern.finditer(text))
    coords = []
    for cm in coord_matches:
        verts = _parse_vertices(cm.group(1))
        coords.append({"pos": cm.end(), "verts": verts})

    index_matches = list(index_pattern.finditer(text))

    meshes = []
    if not coords:
        raise ValueError("No Coordinate { point [...] } block found in VRML file.")

    if len(coords) == 1:
        # single Coordinate block shared by all IndexedFaceSets
        verts_shared = coords[0]["verts"]
        for im in index_matches:
            faces = _parse_faces(im.group(1))
            if faces is not None:
                meshes.append((verts_shared, faces))
    else:
        # multiple Coordinate blocks
        coords_sorted = sorted(coords, key=lambda c: c["pos"])
        coord_idx = 0
        for im in index_matches:
            while (
                coord_idx + 1 < len(coords_sorted)
                and im.start() > coords_sorted[coord_idx + 1]["pos"]
            ):
                coord_idx += 1
            verts = coords_sorted[coord_idx]["verts"]
            faces = _parse_faces(im.group(1))
            if faces is not None:
                meshes.append((verts, faces))

    return meshes


def to_trimesh_list(meshes):
    """
    meshes: list of (vertices, faces) where vertices is the full shared
            Coordinate array; faces index into that array.

    Returns a list of trimesh.Trimesh where each mesh has only the vertices
    actually used by its faces.
    """
    tm_list = []
    for verts, faces in meshes:
        used, inv = np.unique(faces, return_inverse=True)
        verts_local = verts[used]
        faces_local = inv.reshape(faces.shape)
        tm = trimesh.Trimesh(vertices=verts_local, faces=faces_local, process=False)
        tm_list.append(tm)
    return tm_list


def mesh_bbox(tm):
    v = tm.vertices
    return v.min(axis=0), v.max(axis=0)


def fraction_inside_bbox(vertices, bb_min, bb_max):
    inside = np.logical_and(vertices >= bb_min, vertices <= bb_max).all(axis=1)
    return inside.mean()


def assign_cells_to_lineages_strict(cell_tm_list, lineage_bboxes, min_fraction=0.99):
    """
    Assign each cell to the lineage whose bounding box fully contains it.
    """
    assignments = []
    scores = []

    for cell in cell_tm_list:
        v = cell.vertices
        best_idx = -1
        best_score = -1.0
        for i, (bb_min, bb_max) in enumerate(lineage_bboxes):
            frac = fraction_inside_bbox(v, bb_min, bb_max)
            if frac > best_score:
                best_score = frac
                best_idx = i
        if best_score >= min_fraction:
            assignments.append(best_idx)
        else:
            assignments.append(-1)
        scores.append(best_score)

    return np.array(assignments), np.array(scores)


def assign_cells_to_lineages_by_containment(
    cell_tm_list, lineage_meshes, min_fraction=0.60, n_sample=20
):
    """
    Assign each cell to the lineage mesh that contains the most of its volume,
    using ray-casting containment (trimesh.contains) on a small vertex sample.

    A bbox pre-filter skips the ray-cast for lineages whose bbox doesn't
    overlap the cell at all, keeping this fast in practice.

    Parameters
    ----------
    n_sample : int
        Number of vertices to sample per cell for the containment test.
        20 is enough for a reliable fraction estimate while keeping cost low.
    min_fraction : float
        Minimum fraction of sampled vertices that must be inside the lineage
        mesh for the cell to be assigned.
    """
    lineage_bboxes = [mesh_bbox(m) for m in lineage_meshes]
    rng = np.random.default_rng(0)

    assignments = []
    scores = []

    for cell in cell_tm_list:
        v = cell.vertices
        cell_min = v.min(axis=0)
        cell_max = v.max(axis=0)

        # Subsample vertices for the containment test
        if len(v) > n_sample:
            idx = rng.choice(len(v), size=n_sample, replace=False)
            v_sample = v[idx]
        else:
            v_sample = v

        best_idx = -1
        best_score = -1.0

        for i, lin_mesh in enumerate(lineage_meshes):
            bb_min, bb_max = lineage_bboxes[i]

            # Fast bbox pre-filter: skip if cell bbox doesn't overlap lineage bbox
            if np.any(cell_min > bb_max) or np.any(cell_max < bb_min):
                continue

            frac = float(lin_mesh.contains(v_sample).mean())

            if frac > best_score:
                best_score = frac
                best_idx = i

        if best_score >= min_fraction:
            assignments.append(best_idx)
        else:
            assignments.append(-1)
        scores.append(best_score)

    return np.array(assignments), np.array(scores)


# ---------- Slicing + PCA helpers ----------


def mesh_slice_polygon_2d(mesh, plane_origin, plane_normal, u_axis, v_axis):
    """
    Slice a mesh with a plane and return a shapely (Multi)Polygon in
    (PC1, PC2) coordinates, or None if no intersection.
    """
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return None
    polylines = section.discrete
    polys = []
    for pl in polylines:
        if len(pl) < 3:
            continue
        rel = pl - plane_origin
        u = rel @ u_axis
        v = rel @ v_axis
        coords2d = np.column_stack([u, v])
        poly = Polygon(coords2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.area > 0:
            polys.append(poly)
    if not polys:
        return None
    return unary_union(polys)


def project_lineage_polygon_2d(
    lineage_mesh_L,
    plane_origin,
    u_axis,
    v_axis,
    buffer_radius,
    max_points=3000,
):
    """
    Orthographically project the full 3D lineage mesh onto the PC1/PC2 plane
    and build a buffered hull polygon (silhouette).
    """
    verts = lineage_mesh_L.vertices
    rel = verts - plane_origin
    u = rel @ u_axis
    v = rel @ v_axis
    pts2d = np.column_stack([u, v])

    n = len(pts2d)
    if n == 0:
        return None
    if n > max_points:
        stride = int(np.ceil(n / max_points))
        pts2d = pts2d[::stride]

    mp = MultiPoint(pts2d)
    hull = mp.convex_hull
    if hull.is_empty:
        return None

    poly_proj = hull.buffer(buffer_radius).buffer(0)
    if poly_proj.is_empty:
        return None
    return poly_proj


def choose_slice_plane_for_lineage(
    lineage_mesh_L,
    dpn_meshes_L,
    mean_L,
    e1,
    e2,
    e3,
    n_overlap_samples=21,
    n_global_samples=41,
    min_dpn_area=1e-3,
):
    """
    Choose a slice plane that intersects Dpn cells and has large lineage cross-section.
    """
    if not dpn_meshes_L:
        plane_origin = mean_L
        return plane_origin, e3, e1, e2

    dpn_ranges = []
    for m in dpn_meshes_L:
        vrel = m.vertices - mean_L
        s = vrel @ e3
        dpn_ranges.append((s.min(), s.max()))

    s_all_min = max(r[0] for r in dpn_ranges)
    s_all_max = min(r[1] for r in dpn_ranges)

    def lineage_area_at_s(s0):
        origin = mean_L + s0 * e3
        poly_lineage = mesh_slice_polygon_2d(lineage_mesh_L, origin, e3, e1, e2)
        return 0.0 if poly_lineage is None else poly_lineage.area

    def dpn_stats_at_s(s0):
        origin = mean_L + s0 * e3
        count = 0
        for m in dpn_meshes_L:
            poly = mesh_slice_polygon_2d(m, origin, e3, e1, e2)
            if poly is not None and poly.area > min_dpn_area:
                count += 1
        area_lineage = lineage_area_at_s(s0)
        return count, area_lineage

    n_dpns = len(dpn_meshes_L)

    # Case A: overlap band where all Dpns coexist
    if s_all_min <= s_all_max:
        s_candidates = np.linspace(s_all_min, s_all_max, n_overlap_samples)
        best_s = None
        best_area = -1.0
        for s0 in s_candidates:
            count, area = dpn_stats_at_s(s0)
            if count == n_dpns and area > best_area:
                best_area = area
                best_s = s0
        if best_s is not None:
            plane_origin = mean_L + best_s * e3
            return plane_origin, e3, e1, e2

    # Case B: global search
    s_min_global = min(r[0] for r in dpn_ranges)
    s_max_global = max(r[1] for r in dpn_ranges)
    s_candidates = np.linspace(s_min_global, s_max_global, n_global_samples)

    best_s = None
    best_count = -1
    best_area = -1.0
    for s0 in s_candidates:
        count, area = dpn_stats_at_s(s0)
        if (count > best_count) or (count == best_count and area > best_area):
            best_count = count
            best_area = area
            best_s = s0

    plane_origin = mean_L + best_s * e3
    return plane_origin, e3, e1, e2


def lineage_is_connected(tm, min_faces=50, max_secondary_volume=5.0):
    """
    Returns True if the mesh has no problematic disconnected components.

    A secondary component is considered problematic if it has >= min_faces faces
    AND volume >= max_secondary_volume. Small-volume secondary components
    (surface reconstruction artifacts, seam fragments) are tolerated.
    """
    comps = tm.split(only_watertight=False)
    main = max(comps, key=lambda c: len(c.faces))
    secondaries = [c for c in comps if c is not main and len(c.faces) >= min_faces]
    return all(abs(float(c.volume)) < max_secondary_volume for c in secondaries)


def pad_grid_to_canvas(type_grid, extent, ds, canvas_size=800):
    H, W = type_grid.shape
    C = canvas_size
    if H > C or W > C:
        raise ValueError(f"slice {H}x{W} is larger than canvas {C}x{C}")

    canvas_type = np.zeros((C, C), dtype=type_grid.dtype)
    y0 = (C - H) // 2
    x0 = (C - W) // 2
    canvas_type[y0 : y0 + H, x0 : x0 + W] = type_grid

    xmin, xmax, ymin, ymax = extent
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)

    half_width = 0.5 * C * ds
    xmin_C = cx - half_width
    xmax_C = cx + half_width
    ymin_C = cy - half_width
    ymax_C = cy + half_width

    canvas_extent = (xmin_C, xmax_C, ymin_C, ymax_C)
    return canvas_type, canvas_extent


# ---------- Helpers for drawing ----------
def draw_poly_fill(ax, geom, **kwargs):
    """Fill a shapely Polygon/MultiPolygon on a Matplotlib axis."""
    if geom is None:
        return
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        return
    for g in geoms:
        x, y = g.exterior.xy
        ax.fill(x, y, **kwargs)


def draw_poly_outline(ax, geom, **kwargs):
    """Draw only the outline of a shapely Polygon/MultiPolygon."""
    if geom is None:
        return
    if isinstance(geom, Polygon):
        geoms = [geom]
    elif isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
    else:
        return
    for g in geoms:
        x, y = g.exterior.xy
        ax.plot(x, y, **kwargs)
