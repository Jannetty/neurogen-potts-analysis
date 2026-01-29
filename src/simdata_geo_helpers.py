import numpy as np
import re
from pathlib import Path

CELLS_PATTERN = re.compile(r".*_([0-9]{4})_([0-9]{6})\.CELLS\.json$")


# --------------- Dataset and voxel tensor utilities -------------
def build_voxel_tensor_minimal(
    cells_list,
    locs_list,
    grid_size=(800, 800),
):
    """
    Minimal voxelizer for simulation endpoints.

    Returns:
        img : (H, W, 2) float32
            img[..., 0] = NB mask (1 where pop == 1, else 0)
            img[..., 1] = union mask for pops {2, 3}

    Notes:
        - Performs SAME centering logic as the full voxelizer:
          lineage voxels are translated so bounding-box center matches canvas center.
        - No volume channels, no in_lineage channel.
        - Purely a geometric occupancy rasterization.
    """

    W, H = grid_size
    img = np.zeros((H, W, 2), dtype=np.float32)

    # ----------------------------------------------------
    # 1. Collect population labels per cell
    # ----------------------------------------------------
    cell_pop = {}
    for c in cells_list:
        cid = int(c["id"])
        pop = int(c.get("pop", 0))  # expected in {1,2,3}
        cell_pop[cid] = pop

    # ----------------------------------------------------
    # 2. First pass: compute centroid of all voxel coordinates
    # ----------------------------------------------------
    xs, ys = [], []

    for entry in locs_list:
        cid = int(entry["id"])
        if cid not in cell_pop:
            continue

        for region_block in entry.get("location", []):
            for x, y, z in region_block.get("voxels", []):
                if 0 <= x < W and 0 <= y < H:
                    xs.append(x)
                    ys.append(y)

    if not xs:  # no voxels for this lineage
        return img

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    cx_lineage = 0.5 * (min_x + max_x)
    cy_lineage = 0.5 * (min_y + max_y)

    cx_canvas = (W - 1) / 2.0
    cy_canvas = (H - 1) / 2.0

    dx = int(round(cx_canvas - cx_lineage))
    dy = int(round(cy_canvas - cy_lineage))

    # ----------------------------------------------------
    # 3. Second pass: paint masks
    # ----------------------------------------------------
    for entry in locs_list:
        cid = int(entry["id"])
        pop = cell_pop.get(cid)
        if pop is None:
            continue

        is_NB = pop == 1
        is_union = pop in (2, 3)

        for region_block in entry.get("location", []):
            for x, y, z in region_block.get("voxels", []):
                xs = x + dx
                ys = y + dy
                if 0 <= xs < W and 0 <= ys < H:
                    if is_NB:
                        img[ys, xs, 0] = 1.0
                    if is_union:
                        img[ys, xs, 1] = 1.0

    return img


def index_time_series_files(base_path: Path, model_runs):
    """
    CELLS and LOCATIONS live together in:
      base_path / mr / *.CELLS.json
      base_path / mr / *.LOCATIONS.json
    """
    index = {}

    for mr in model_runs:
        run_dir = base_path / mr
        if not run_dir.exists():
            print(f"[index] Skipping {mr}: missing dir {run_dir}")
            continue

        for cells_path in run_dir.glob("*.CELLS.json"):
            m = CELLS_PATTERN.match(cells_path.name)
            if not m:
                continue

            run_id, time_id = m.group(1), m.group(2)

            locs_path = cells_path.with_name(
                cells_path.name.replace(".CELLS.json", ".LOCATIONS.json")
            )
            if not locs_path.exists():
                # Optional: print once in a while if debugging
                # print(f"[index] Missing LOCATIONS for {cells_path.name}")
                continue

            key = (mr, run_id)
            entry = (int(time_id), cells_path, locs_path)
            index.setdefault(key, []).append(entry)

    # sort by time
    for key in index:
        index[key].sort(key=lambda x: x[0])

    return index
