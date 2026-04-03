"""
compare_detdiff.py

Compares mud mutant simulation outputs with and without imposed deterministic
differentiation dynamics.

  No det-diff (rule-based):  sim11-20  (VOLUME_BASED_CRITICAL_VOLUME=1)
                              sim31-40  (VOLUME_BASED_CRITICAL_VOLUME=0)
  Det-diff (deterministic):  sim41-50  (VOLUME_BASED_CRITICAL_VOLUME=1)
                              sim51-60  (VOLUME_BASED_CRITICAL_VOLUME=0)

Each pair (e.g. sim11 vs sim41) shares all parameters except HAS_DETERMINISTIC_DIFFERENTIATION.

Metrics computed per replicate from endpoint (last timepoint) CELLS files:
  - NB count          (pop=1)
  - GMC count         (pop=2)
  - Neuron count      (pop=3)
  - Total cell count
  - NB fraction       (NB count / total)
  - Mean NB area      (mean NB voxels * ds^2,  µm²)
  - Lineage area      (total voxels * ds^2,    µm²)
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DS = 0.3          # µm / voxel side
DS2 = DS ** 2     # µm² per voxel (2D projected area)

OUTPUT_BASE = os.path.expanduser("~/bagherilab/neurogen-potts-analysis/data/sim")
PLOT_DIR    = os.path.expanduser("~/bagherilab/neurogen-potts-analysis/data/sim/bioparams_analysis")
os.makedirs(PLOT_DIR, exist_ok=True)

# Paired sim numbers and their human-readable labels
# (no_detdiff_sim, detdiff_sim, label, ruleset, crit_vol_flag)
SIM_PAIRS = [
    (11, 41, "volume_none",        "volume",   1),
    (12, 42, "volume_nbContactABM","volume",   1),
    (13, 43, "volume_volumeABM",   "volume",   1),
    (14, 44, "volume_nbContactPDE","volume",   1),
    (15, 45, "volume_volumePDE",   "volume",   1),
    (16, 46, "location_none",      "location", 1),
    (17, 47, "location_nbContactABM","location",1),
    (18, 48, "location_volumeABM", "location", 1),
    (19, 49, "location_nbContactPDE","location",1),
    (20, 50, "location_volumePDE", "location", 1),
    (31, 51, "volume_none",        "volume",   0),
    (32, 52, "volume_nbContactABM","volume",   0),
    (33, 53, "volume_volumeABM",   "volume",   0),
    (34, 54, "volume_nbContactPDE","volume",   0),
    (35, 55, "volume_volumePDE",   "volume",   0),
    (36, 56, "location_none",      "location", 0),
    (37, 57, "location_nbContactABM","location",0),
    (38, 58, "location_volumeABM", "location", 0),
    (39, 59, "location_nbContactPDE","location",0),
    (40, 60, "location_volumePDE", "location", 0),
]

METRICS = ["nb_count", "gmc_count", "neuron_count", "total_count",
           "nb_fraction", "mean_nb_area", "lineage_area"]
METRIC_LABELS = {
    "nb_count":      "NB count",
    "gmc_count":     "GMC count",
    "neuron_count":  "Neuron count",
    "total_count":   "Total cell count",
    "nb_fraction":   "NB fraction",
    "mean_nb_area":  "Mean NB projected area (µm²)",
    "lineage_area":  "Lineage projected area (µm²)",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_sim_metrics(sim_num):
    """Return a dict of metric -> np.array across all replicates for one sim."""
    sim_dir = os.path.join(OUTPUT_BASE, f"sim{sim_num:02d}")
    all_cells_files = sorted(glob.glob(os.path.join(sim_dir, "*.CELLS.json")))
    if not all_cells_files:
        raise FileNotFoundError(f"No CELLS files found for sim{sim_num:02d} at {sim_dir}")
    last_timepoint = max(
        os.path.basename(f).rsplit("_", 1)[-1].replace(".CELLS.json", "")
        for f in all_cells_files
        if not f.endswith("_000000.CELLS.json")
    )
    pattern = os.path.join(sim_dir, f"*_{last_timepoint}.CELLS.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No endpoint CELLS files found for sim{sim_num:02d} at {pattern}")

    records = {m: [] for m in METRICS}
    for fpath in files:
        with open(fpath) as f:
            cells = json.load(f)

        nb_voxels  = [c["voxels"] for c in cells if c["pop"] == 1]
        gmc_cells  = [c for c in cells if c["pop"] == 2]
        neu_cells  = [c for c in cells if c["pop"] == 3]

        nb_count    = len(nb_voxels)
        gmc_count   = len(gmc_cells)
        neuron_count = len(neu_cells)
        total_count  = nb_count + gmc_count + neuron_count

        records["nb_count"].append(nb_count)
        records["gmc_count"].append(gmc_count)
        records["neuron_count"].append(neuron_count)
        records["total_count"].append(total_count)
        records["nb_fraction"].append(nb_count / total_count if total_count > 0 else 0)
        records["mean_nb_area"].append(np.mean(nb_voxels) * DS2 if nb_voxels else 0)
        records["lineage_area"].append(sum(c["voxels"] for c in cells) * DS2)

    return {m: np.array(v) for m, v in records.items()}


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
def print_summary(all_data):
    header = f"{'Sim pair':<12} {'Label':<30} {'ruleset':<10} {'critVol':>7} | " + \
             " | ".join(f"{METRIC_LABELS[m][:18]:>20}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for (no_dd, dd, label, ruleset, cv) in SIM_PAIRS:
        for tag, snum in [("no-detdiff", no_dd), ("detdiff   ", dd)]:
            d = all_data[(snum, tag.strip())]
            vals = "  |  ".join(f"{np.mean(d[m]):>8.2f} ± {np.std(d[m]):>5.2f}" for m in METRICS)
            print(f"sim{snum:02d} {tag}  {label:<30} {ruleset:<10} {cv:>7} | {vals}")
        print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
COLORS = {"no-detdiff": "#4C72B0", "detdiff": "#DD8452"}

def plot_metric_comparison(all_data, metric):
    """One figure: 20 paired bar groups (one per sim pair), each with no-detdiff vs detdiff."""
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharey=False)

    for ax_idx, cv in enumerate([1, 0]):
        ax = axes[ax_idx]
        pairs = [(no_dd, dd, label, ruleset) for no_dd, dd, label, rs, c in SIM_PAIRS if c == cv
                 for ruleset in [rs]]
        # rebuild cleanly
        pairs = [(no_dd, dd, label, ruleset) for (no_dd, dd, label, ruleset, c) in SIM_PAIRS if c == cv]

        x = np.arange(len(pairs))
        width = 0.35

        for i, (no_dd, dd, label, _) in enumerate(pairs):
            nd = all_data[(no_dd, "no-detdiff")][metric]
            dt = all_data[(dd,    "detdiff")]   [metric]
            ax.bar(x[i] - width/2, np.mean(nd), width, yerr=np.std(nd),
                   color=COLORS["no-detdiff"], capsize=3, alpha=0.85)
            ax.bar(x[i] + width/2, np.mean(dt), width, yerr=np.std(dt),
                   color=COLORS["detdiff"],    capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, _, label, _ in pairs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"VOLUME_BASED_CRITICAL_VOLUME = {cv}")
        ax.axhline(0, color="black", linewidth=0.5)

    patch_nd = mpatches.Patch(color=COLORS["no-detdiff"], label="No det-diff (rule-based)")
    patch_dd = mpatches.Patch(color=COLORS["detdiff"],    label="Det-diff (deterministic)")
    fig.legend(handles=[patch_nd, patch_dd], loc="upper right", fontsize=10)
    fig.suptitle(f"Mud mutant comparison: {METRIC_LABELS[metric]}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = os.path.join(PLOT_DIR, f"compare_detdiff_{metric}.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def plot_nb_fraction_scatter(all_data):
    """Scatter: no-detdiff NB fraction vs detdiff NB fraction, one point per sim pair."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax_idx, cv in enumerate([1, 0]):
        ax = axes[ax_idx]
        pairs = [(no_dd, dd, label, ruleset) for (no_dd, dd, label, ruleset, c) in SIM_PAIRS if c == cv]
        for no_dd, dd, label, ruleset in pairs:
            nd_mean = np.mean(all_data[(no_dd, "no-detdiff")]["nb_fraction"])
            dt_mean = np.mean(all_data[(dd,    "detdiff")]   ["nb_fraction"])
            color = "#4C72B0" if ruleset == "volume" else "#55A868"
            ax.scatter(nd_mean, dt_mean, color=color, s=60, zorder=3)
            ax.annotate(label, (nd_mean, dt_mean), fontsize=6, ha="left",
                        xytext=(3, 3), textcoords="offset points")

        lims = [ax.get_xlim(), ax.get_ylim()]
        lo = min(lims[0][0], lims[1][0])
        hi = max(lims[0][1], lims[1][1])
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x")
        ax.set_xlabel("NB fraction — no det-diff")
        ax.set_ylabel("NB fraction — det-diff")
        ax.set_title(f"VOLUME_BASED_CRITICAL_VOLUME = {cv}")

        vol_patch = mpatches.Patch(color="#4C72B0", label="volume ruleset")
        loc_patch = mpatches.Patch(color="#55A868", label="location ruleset")
        ax.legend(handles=[vol_patch, loc_patch], fontsize=8)

    fig.suptitle("NB fraction: rule-based vs deterministic differentiation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    outpath = os.path.join(PLOT_DIR, "compare_detdiff_nb_fraction_scatter.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading simulation data...")
    all_data = {}
    for no_dd, dd, label, ruleset, cv in SIM_PAIRS:
        all_data[(no_dd, "no-detdiff")] = load_sim_metrics(no_dd)
        all_data[(dd,    "detdiff")]    = load_sim_metrics(dd)
    print(f"Loaded {len(all_data)} sim datasets.\n")

    print("=== Summary (mean ± SD across 50 replicates) ===\n")
    print_summary(all_data)

    print("\nGenerating plots...")
    for metric in METRICS:
        plot_metric_comparison(all_data, metric)
    plot_nb_fraction_scatter(all_data)

    print("\nDone. Plots saved to:", PLOT_DIR)
