# analyze.pyfrom __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from scipy.stats import mannwhitneyu
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BASE = Path("data")

SIM_NPZ = BASE / "sim" / "sim_geo_counts_endpoints.npz"

EXP_WT_DIR = BASE / "exp" / "npz_files" / "exp_wt_npz"
EXP_MUD_DIR = BASE / "exp" / "npz_files" / "exp_mud_npz"

EXP_WT_NPZS = sorted(EXP_WT_DIR.glob("*.npz")) if EXP_WT_DIR.exists() else []
EXP_MUD_NPZS = sorted(EXP_MUD_DIR.glob("*.npz")) if EXP_MUD_DIR.exists() else []

# If you only want sim condition 01..20, set these explicitly:
SIM_CONDITIONS = [f"{i:02d}" for i in range(1, 21)]

# Toggle plotting (histograms per metric for each sim condition)
PLOT_HISTS = False

# Mann–Whitney settings
MWU_ALTERNATIVE = "two-sided"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def safe_load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return dict(np.load(path, allow_pickle=True))


def compute_sizes_from_geo(geo: np.ndarray) -> dict:
    """
    geo: (N,H,W,2) where:
      geo[...,0] = dpn mask
      geo[...,1] = union mask for pops {2,3} but may overlap dpn

    Returns per-lineage voxel-size proxies:
      dpn_sizes, pros_only_sizes, lineage_sizes
    """
    assert geo.ndim == 4 and geo.shape[-1] == 2, f"Unexpected geo shape: {geo.shape}"
    N = geo.shape[0]

    dpn = geo[..., 0] > 0.5
    union_raw = geo[..., 1] > 0.5
    pros_only = union_raw & (~dpn)
    lineage = dpn | pros_only

    dpn_sizes = dpn.reshape(N, -1).sum(axis=1).astype(np.float64)
    pros_only_sizes = pros_only.reshape(N, -1).sum(axis=1).astype(np.float64)
    lineage_sizes = lineage.reshape(N, -1).sum(axis=1).astype(np.float64)

    return {
        "dpn_sizes": dpn_sizes,
        "pros_only_sizes": pros_only_sizes,
        "lineage_sizes": lineage_sizes,
    }


def compute_metrics(geo: np.ndarray, counts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    counts: (N,2) where:
      counts[:,0] = dpn cell count
      counts[:,1] = pros-like cell count (pops 2+3)
    """
    assert counts.ndim == 2 and counts.shape[1] == 2, f"Unexpected counts shape: {counts.shape}"

    sizes = compute_sizes_from_geo(geo)

    dpn_counts = counts[:, 0].astype(np.float64)
    pros_counts = counts[:, 1].astype(np.float64)
    total_counts = dpn_counts + pros_counts

    lineage_volumes = sizes["lineage_sizes"]
    dpn_volumes = sizes["dpn_sizes"]

    avg_nb_volume = dpn_volumes / np.maximum(dpn_counts, 1.0)  # per-lineage mean NB size proxy
    avg_lineage_volume_overall = lineage_volumes               # per-lineage lineage size proxy

    return {
        # what you asked for (distributions)
        "cell_counts_total": total_counts,
        "lineage_volumes": lineage_volumes,
        "avg_nb_volume": avg_nb_volume,
        "avg_lineage_volume_overall": avg_lineage_volume_overall,
    }


def summarize(x: np.ndarray) -> Tuple[int, float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (0, float("nan"), float("nan"))
    return (int(x.size), float(np.mean(x)), float(np.median(x)))


def mwu_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if (not HAVE_SCIPY) or a.size < 2 or b.size < 2:
        return float("nan")
    _, p = mannwhitneyu(a, b, alternative=MWU_ALTERNATIVE)
    return float(p)


def pooled_exp_metrics(npz_paths: List[Path]) -> Dict[str, np.ndarray]:
    """
    Pool ALL experimental lobes into one big distribution per metric.
    """
    all_geo = []
    all_counts = []
    for p in npz_paths:
        d = safe_load_npz(p)
        if "geo" not in d or "counts" not in d:
            print(f"[exp] skipping {p.name}: missing geo/counts")
            continue
        all_geo.append(d["geo"])
        all_counts.append(d["counts"])

    if not all_geo:
        return {k: np.array([]) for k in ["cell_counts_total", "lineage_volumes", "avg_nb_volume", "avg_lineage_volume_overall"]}

    geo = np.concatenate(all_geo, axis=0)
    counts = np.concatenate(all_counts, axis=0)
    return compute_metrics(geo, counts)


def sim_metrics_by_condition(sim_npz_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return metrics per sim condition based on model_run.
    model_run is saved as strings like "01", "02", ... (or maybe ints).
    """
    d = safe_load_npz(sim_npz_path)
    geo = d["geo"]
    counts = d["counts"]
    model_run = d["model_run"]

    # normalize model_run to 2-digit strings
    model_run_str = np.array([f"{int(x):02d}" if str(x).isdigit() else str(x) for x in model_run], dtype=object)

    out = {}
    for cond in np.unique(model_run_str):
        mask = model_run_str == cond
        out[cond] = compute_metrics(geo[mask], counts[mask])
    return out


def plot_hist_overlay(sim_vals: np.ndarray, exp_vals: np.ndarray, title: str, xlabel: str):
    plt.figure(figsize=(7, 5))
    # bins chosen from combined range
    allv = np.concatenate([sim_vals, exp_vals]).astype(float)
    if allv.size == 0:
        return
    vmin, vmax = float(np.min(allv)), float(np.max(allv))
    if vmin == vmax:
        vmin -= 1
        vmax += 1
    bins = np.linspace(vmin, vmax, 40)

    plt.hist(sim_vals, bins=bins, alpha=0.4, label="sim", edgecolor="black")
    plt.hist(exp_vals, bins=bins, alpha=0.4, label="exp", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    if not SIM_NPZ.exists():
        raise FileNotFoundError(f"Simulation NPZ not found: {SIM_NPZ}")

    if not EXP_WT_NPZS:
        print(f"[warn] No WT experimental NPZs found in {EXP_WT_DIR}")
    if not EXP_MUD_NPZS:
        print(f"[warn] No Mud experimental NPZs found in {EXP_MUD_DIR}")

    exp_wt = pooled_exp_metrics(EXP_WT_NPZS)
    exp_mud = pooled_exp_metrics(EXP_MUD_NPZS)

    sim_by_cond = sim_metrics_by_condition(SIM_NPZ)

    metrics_to_compare = [
        ("cell_counts_total", "Total cell count per lineage", "# cells"),
        ("lineage_volumes", "Lineage volume per lineage (voxels)", "# voxels (dpn ∪ pros-only)"),
        ("avg_nb_volume", "Average neuroblast volume (per lineage)", "dpn_voxels / dpn_cell_count"),
        ("avg_lineage_volume_overall", "Average lineage volume overall (per lineage)", "# voxels (dpn ∪ pros-only)"),
    ]

    if not HAVE_SCIPY:
        print("[warn] scipy not available; p-values will be NaN")

    print("\n==================== Sim condition vs Experimental (WT / Mud) ====================")
    for cond in SIM_CONDITIONS:
        if cond not in sim_by_cond:
            print(f"\n[sim {cond}] missing from sim NPZ, skipping.")
            continue

        simm = sim_by_cond[cond]
        print(f"\n--- SIM CONDITION {cond} ---")

        for key, title, xlabel in metrics_to_compare:
            sim_vals = np.asarray(simm[key], dtype=float)

            wt_vals = np.asarray(exp_wt[key], dtype=float)
            mud_vals = np.asarray(exp_mud[key], dtype=float)

            n_s, mean_s, med_s = summarize(sim_vals)

            n_w, mean_w, med_w = summarize(wt_vals)
            n_m, mean_m, med_m = summarize(mud_vals)

            p_wt = mwu_pvalue(sim_vals, wt_vals)
            p_mud = mwu_pvalue(sim_vals, mud_vals)

            print(f"\n  Metric: {key}")
            print(f"    sim: N={n_s:5d} mean={mean_s:10.3f} median={med_s:10.3f}")
            print(f"    WT : N={n_w:5d} mean={mean_w:10.3f} median={med_w:10.3f}  p(sim vs WT) ={p_wt:.3e}")
            print(f"    Mud: N={n_m:5d} mean={mean_m:10.3f} median={med_m:10.3f}  p(sim vs Mud)={p_mud:.3e}")

            if PLOT_HISTS and wt_vals.size > 0:
                plot_hist_overlay(sim_vals, wt_vals, f"Sim {cond} vs WT exp — {title}", xlabel)
            if PLOT_HISTS and mud_vals.size > 0:
                plot_hist_overlay(sim_vals, mud_vals, f"Sim {cond} vs Mud exp — {title}", xlabel)

    print("\n==================================================================================\n")