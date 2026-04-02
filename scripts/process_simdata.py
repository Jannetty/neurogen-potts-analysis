from pathlib import Path
import numpy as np
import json

import random
import matplotlib.pyplot as plt

from src.simdata_geo_helpers import build_voxel_tensor_minimal, index_time_series_files


def infer_condition_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "mud" in name:
        return "mud"
    if "wt" in name:
        return "wt"
    return "unknown"


def precompute_sim_geo_counts_to_npz(
    base_path: Path,
    model_runs,
    grid_size=(800, 800),
    out_npz_path=None,
):
    """
    For each (model_run, run_id):
      - identify LAST timepoint (as in precompute_endpoints_to_disk)
      - load its CELLS + LOCATIONS
      - convert to (H, W, 2) geo tensor via build_voxel_tensor_minimal
      - collect scalar counts [N_dpn_like, N_pros_like]

    Save a single .npz with:
      geo: (N_sims, H, W, 2)
      counts: (N_sims, 2)      # [N_dpn_like, N_pros_like]
      model_run: (N_sims,)
      run_id: (N_sims,)
      time_id: (N_sims,)
      grid_size: (2,)
    """

    if out_npz_path is None:
        out_npz_path = base_path / "sim_geo_counts_endpoints.npz"
    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    index = index_time_series_files(base_path, model_runs)
    print(f"[sim_geo_counts] Found {len(index)} simulations total.")

    geo_list = []
    counts_list = []
    model_run_list = []
    run_id_list = []
    time_id_list = []
    condition_list = []

    for (mr, run_id), entries in sorted(index.items()):
        if not entries:
            continue

        # last entry is final timepoint after sorting
        time_id, cells_path, locs_path = entries[-1]
        condition = infer_condition_from_filename(cells_path)
        condition_list.append(condition)

        print(f"[sim_geo_counts] {mr} run {run_id}: last timepoint {time_id}")

        # load files
        with open(cells_path, "r") as f:
            cells_list = json.load(f)
        with open(locs_path, "r") as f:
            locs_list = json.load(f)

        # geometry tensor (H, W, 2)
        geo = build_voxel_tensor_minimal(
            cells_list,
            locs_list,
            grid_size=grid_size,
        )

        # scalar counts
        pops = [int(c["pop"]) for c in cells_list]
        N_dpn_like = sum(p == 1 for p in pops)
        N_pros_like = sum(p in (2, 3) for p in pops)
        counts = np.array([N_dpn_like, N_pros_like], dtype=np.int32)

        geo_list.append(geo)
        counts_list.append(counts)
        model_run_list.append(mr)
        run_id_list.append(run_id)
        time_id_list.append(time_id)

    if not geo_list:
        raise RuntimeError(
            "[sim_geo_counts] No simulations processed; nothing to save."
        )

    geo_array = np.stack(geo_list, axis=0)  # (N_sims, H, W, 2)
    counts_array = np.stack(counts_list, axis=0)  # (N_sims, 2)
    model_run_arr = np.array(model_run_list)
    run_id_arr = np.array(run_id_list)
    time_id_arr = np.array(time_id_list)

    print(f"[sim_geo_counts] Final shapes:")
    print("  geo:", geo_array.shape)
    print("  counts:", counts_array.shape)

    condition_arr = np.array(condition_list)

    np.savez(
        out_npz_path,
        geo=geo_array,
        counts=counts_array,
        model_run=model_run_arr,
        run_id=run_id_arr,
        time_id=time_id_arr,
        condition=condition_arr,
        grid_size=np.array(grid_size, dtype=int),
    )

    print(f"[sim_geo_counts] Saved → {out_npz_path}")
    return out_npz_path


# --------- Sanity Checks --------------
def show_sim_sample(geo, counts, sample_idx, title_prefix="Sim"):
    """
    geo: (N, H, W, 2)
    counts: (N, 2) -> [N_dpn_like, N_pros_like]
    """
    img = geo[sample_idx]
    N_dpn, N_pros = counts[sample_idx]

    print(f"\n--- {title_prefix} (sample idx={sample_idx}) ---")
    print(f"  N_dpn_like (pop==1):   {int(N_dpn)}")
    print(f"  N_pros_like (pop 2+3): {int(N_pros)}")

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img[..., 0], cmap="Purples")
    axs[0].set_title(f"{title_prefix}\nNB-like mask\n(N_dpn_like={int(N_dpn)})")
    axs[0].axis("off")

    axs[1].imshow(img[..., 1], cmap="Greys")
    axs[1].set_title(f"{title_prefix}\nunion (pop2+3)\n(N_pros_like={int(N_pros)})")
    axs[1].axis("off")

    overlay = img[..., 1].copy()
    overlay[img[..., 0] > 0] = 2
    axs[2].imshow(overlay, cmap="viridis")
    axs[2].set_title("Overlay (NB overwrites union)")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()


def check_sim_values(name, geo):
    print(f"\nValue check for {name}:")
    print("  min:", geo.min(), "max:", geo.max())
    unique_vals = np.unique(geo)
    print(f"  unique values (first 10): {unique_vals[:10]}")
    if not np.all((geo == 0) | (geo == 1)):
        print("⚠️  WARNING: non-binary values detected")


def check_counts_vs_masks(geo, counts, max_examples=20):
    """
    Basic logical consistency:
      - If N_dpn_like == 0  → NB mask must be empty
      - If N_dpn_like > 0   → NB mask must have at least 1 voxel
      - Similarly for union mask and N_pros_like
    """
    N = geo.shape[0]
    n_checked = min(N, max_examples)

    print(f"\nChecking counts vs masks on {n_checked} examples...")

    for i in range(n_checked):
        img = geo[i]
        N_dpn, N_pros = counts[i]

        nb_mask = img[..., 0] > 0.5
        union_mask = img[..., 1] > 0.5

        nb_voxels = int(nb_mask.sum())
        union_voxels = int(union_mask.sum())

        if N_dpn == 0 and nb_voxels > 0:
            print(f"⚠️ sample {i}: N_dpn_like=0 but NB mask has {nb_voxels} voxels")
        if N_dpn > 0 and nb_voxels == 0:
            print(f"⚠️ sample {i}: N_dpn_like={N_dpn} but NB mask has 0 voxels")

        if N_pros == 0 and union_voxels > 0:
            print(
                f"⚠️ sample {i}: N_pros_like=0 but union mask has {union_voxels} voxels"
            )
        if N_pros > 0 and union_voxels == 0:
            print(f"⚠️ sample {i}: N_pros_like={N_pros} but union mask has 0 voxels")

    print("Counts vs masks check done.")


def sanity_check_sim_npz(npz_path):
    print("\nLoading sim NPZ:", npz_path)
    data = np.load(npz_path, allow_pickle=True)

    geo = data["geo"]  # (N, H, W, 2)
    counts = data["counts"]  # (N, 2)
    model_run = data["model_run"]
    run_id = data["run_id"]
    time_id = data["time_id"]
    grid_size = data["grid_size"]
    condition = data["condition"]  # (N,)

    print("Shapes:")
    print("  geo:", geo.shape)
    print("  counts:", counts.shape)
    print("  model_run:", model_run.shape)
    print("  run_id:", run_id.shape)
    print("  time_id:", time_id.shape)
    print("  condition:", condition.shape)
    print("  grid_size:", grid_size)

    check_sim_values("sim", geo)

    # ----- categories based on condition -----
    cond_str = np.array([str(c).lower() for c in condition])

    mud_mask = cond_str == "mud"
    wt_mask = cond_str == "wt"
    unk_mask = ~(mud_mask | wt_mask)

    wt_indices = np.where(wt_mask)[0]
    mud_indices = np.where(mud_mask)[0]
    unk_indices = np.where(unk_mask)[0]

    print("\nCondition categories:")
    print(f"  WT entries:      {len(wt_indices)}")
    print(f"  mud entries:     {len(mud_indices)}")
    print(f"  unknown entries: {len(unk_indices)}")

    # Show one WT-like example if available
    if len(wt_indices) > 0:
        idx_wt = int(np.random.choice(wt_indices))
        title_prefix = (
            f"WT mr={model_run[idx_wt]} run={run_id[idx_wt]} t={time_id[idx_wt]}"
        )
        show_sim_sample(geo, counts, idx_wt, title_prefix=title_prefix)
    else:
        print("⚠️ No WT entries detected (condition=='wt').")

    # Show one mudmut-like example if available
    if len(mud_indices) > 0:
        idx_mud = int(np.random.choice(mud_indices))
        title_prefix = (
            f"Mud mr={model_run[idx_mud]} run={run_id[idx_mud]} t={time_id[idx_mud]}"
        )
        show_sim_sample(geo, counts, idx_mud, title_prefix=title_prefix)
    else:
        print("⚠️ No mud entries detected (condition=='mud').")

    # Optionally: a couple extra random examples from the full set
    print("\nPlotting a couple of additional random simulations...")
    for _ in range(2):
        idx = random_sample_idx(geo)
        title_prefix = f"Random cond={condition[idx]} mr={model_run[idx]} run={run_id[idx]} t={time_id[idx]}"
        show_sim_sample(geo, counts, idx, title_prefix=title_prefix)

    check_counts_vs_masks(geo, counts)

    print("\nSimulation NPZ sanity check complete.\n")


def random_sample_idx(geo):
    return random.randint(0, geo.shape[0] - 1)


if __name__ == "__main__":
    base = Path("data/sim")
    # MODEL_RUNS = [p.name for p in base.iterdir() if p.is_dir()]
    MODEL_RUNS = [f"sim{i:02d}" for i in range(1, 41)]
    print(f"Detected model runs: {MODEL_RUNS}")
    out_npz = precompute_sim_geo_counts_to_npz(base, MODEL_RUNS, grid_size=(200, 200))
    sanity_check_sim_npz(out_npz)
