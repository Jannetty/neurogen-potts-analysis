from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal


# -------------------------------
# Config
# -------------------------------
SIM_NPZ = Path("data/sim/sim_geo_counts_endpoints.npz")
SIM_CONDITIONS = [f"{i:02d}" for i in range(1, 41)]
EXP_NPZ_DIRS = {
    "wt": Path("data/exp/npz_files/exp_wt_npz"),
    "mud": Path("data/exp/npz_files/exp_mud_npz"),
    "nanobody": Path("data/exp/npz_files/exp_nanobody_npz"),
}

# one plot per genotype per metric
GENOTYPES = ["wt", "mud"]

# metrics of interest (keys -> pretty title, x label)
METRICS = {
    "cell_counts_total": ("Total cell counts per lineage", "cells per lineage"),
    "lineage_volumes": ("Lineage volumes per lineage", "voxels (dpn ∪ pros-only)"),
    "avg_nb_volume": ("Average neuroblast volume per lineage", "dpn_voxels / dpn_cell_count"),
    "avg_lineage_volume": ("Average lineage volume per lineage", "voxels (dpn ∪ pros-only)"),
}

EXP_COMPARE_METRICS = ["cell_counts_total", "lineage_volumes", "avg_nb_volume"]

# ----------------------------------------------
# Helper functions for significance calculation
# ----------------------------------------------

def significance_star(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def bonferroni_correct(pvals):
    n = len(pvals)
    return [min(p * n, 1.0) for p in pvals]


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return np.nan
    return (2.0 * float(u_stat) / float(n1 * n2)) - 1.0

# -------------------------------
# Metric computation
# -------------------------------
def compute_sizes_from_geo(geo: np.ndarray) -> Dict[str, np.ndarray]:
    """
    geo: (N,H,W,2)
      channel0 = dpn mask
      channel1 = union mask for pops {2,3} but may overlap dpn
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

    return {"dpn_sizes": dpn_sizes, "pros_only_sizes": pros_only_sizes, "lineage_sizes": lineage_sizes}


def compute_metrics(geo: np.ndarray, counts: np.ndarray) -> Dict[str, np.ndarray]:
    assert counts.ndim == 2 and counts.shape[1] == 2, f"Unexpected counts shape: {counts.shape}"

    sizes = compute_sizes_from_geo(geo)
    dpn_counts = counts[:, 0].astype(np.float64)
    pros_counts = counts[:, 1].astype(np.float64)

    total_counts = dpn_counts + pros_counts
    lineage_volumes = sizes["lineage_sizes"]
    avg_nb_volume = sizes["dpn_sizes"] / np.maximum(dpn_counts, 1.0)

    # NOTE: this is the same as lineage_volumes; kept separate for clarity
    avg_lineage_volume = lineage_volumes

    return {
        "cell_counts_total": total_counts,
        "lineage_volumes": lineage_volumes,
        "avg_nb_volume": avg_nb_volume,
        "avg_lineage_volume": avg_lineage_volume,
    }


# -------------------------------
# Loading + grouping
# -------------------------------
def load_sim_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    geo = data["geo"]
    counts = data["counts"]
    model_run = data["model_run"]
    condition = data.get("condition", None)
    if condition is None:
        raise KeyError("NPZ is missing 'condition'. Re-save sim NPZ with condition included.")
    return geo, counts, model_run, condition


def load_experimental_arrays(npz_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not npz_dir.exists():
        raise FileNotFoundError(f"Experimental NPZ directory not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in experimental directory: {npz_dir}")

    geo_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        if "geo" not in data or "counts" not in data:
            raise KeyError(f"{fpath} is missing required keys 'geo' and/or 'counts'.")
        geo_list.append(data["geo"])
        counts_list.append(data["counts"])

    geo = np.concatenate(geo_list, axis=0)
    counts = np.concatenate(counts_list, axis=0)
    return geo, counts


def experimental_metrics_by_genotype() -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for geno, exp_dir in EXP_NPZ_DIRS.items():
        geo, counts = load_experimental_arrays(exp_dir)
        out[geno] = compute_metrics(geo, counts)
    return out


def normalize_model_run(model_run: np.ndarray) -> np.ndarray:
    # handles ints or strings
    out = []
    for x in model_run:
        s = str(x)
        if s.isdigit():
            out.append(f"{int(s):02d}")
        else:
            # if it's already like "01"
            try:
                out.append(f"{int(float(s)):02d}")
            except Exception:
                out.append(s)
    return np.array(out, dtype=object)


def normalize_condition(cond: np.ndarray) -> np.ndarray:
    return np.array([str(c).lower() for c in cond], dtype=object)


def values_by_genotype_and_condition(
    geo: np.ndarray,
    counts: np.ndarray,
    model_run: np.ndarray,
    condition: np.ndarray,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """
    Returns:
      out[genotype][cond][metric] = 1D array (per-lineage values)
    """
    model_run = normalize_model_run(model_run)
    condition = normalize_condition(condition)

    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {g: {} for g in GENOTYPES}

    for geno in GENOTYPES:
        for cond in SIM_CONDITIONS:
            mask = (condition == geno) & (model_run == cond)
            if not np.any(mask):
                continue
            m = compute_metrics(geo[mask], counts[mask])
            out[geno][cond] = m

    return out


# -------------------------------
# Plotting
# -------------------------------
def print_summary_table(vals: Dict[str, Dict[str, Dict[str, np.ndarray]]]):
    print("\n=== Summary (mean ± std; N) per genotype x condition ===")
    for geno in GENOTYPES:
        if not vals[geno]:
            print(f"\n{geno.upper()}: (no data)")
            continue
        print(f"\n{geno.upper()}:")
        for cond in SIM_CONDITIONS:
            if cond not in vals[geno]:
                continue
            print(f"  condition {cond}:")
            for key in ["cell_counts_total", "lineage_volumes", "avg_nb_volume", "avg_lineage_volume"]:
                arr = np.asarray(vals[geno][cond][key], dtype=float)
                if arr.size == 0:
                    continue
                print(f"    {key:18s}  mean={arr.mean():.3f}  std={arr.std():.3f}  N={arr.size:d}")
    print("========================================================\n")


def print_experimental_summary_table(exp_metrics: Dict[str, Dict[str, np.ndarray]]):
    print("\n=== Experimental Summary (aggregate over all lineages) ===")
    for geno in EXP_NPZ_DIRS:
        if geno not in exp_metrics:
            print(f"\n{geno.upper()}: (no data)")
            continue

        nb_arr = np.asarray(exp_metrics[geno]["avg_nb_volume"], dtype=float)
        count_arr = np.asarray(exp_metrics[geno]["cell_counts_total"], dtype=float)
        lineage_arr = np.asarray(exp_metrics[geno]["avg_lineage_volume"], dtype=float)

        print(f"\n{geno.upper()} experimental group:")
        print(
            "  average neuroblast volume "
            f"(voxels / dpn cell): mean={nb_arr.mean():.3f} std={nb_arr.std():.3f} N={nb_arr.size:d}"
        )
        print(
            "  average cell count "
            f"(cells / lineage): mean={count_arr.mean():.3f} std={count_arr.std():.3f} N={count_arr.size:d}"
        )
        print(
            "  average lineage volume "
            f"(voxels / lineage): mean={lineage_arr.mean():.3f} std={lineage_arr.std():.3f} N={lineage_arr.size:d}"
        )
    print("=======================================================\n")


def print_simulation_summary_table(vals: Dict[str, Dict[str, Dict[str, np.ndarray]]]):
    print("\n=== Simulation Summary (aggregate over all lineages) ===")
    for geno in EXP_NPZ_DIRS:
        if geno not in vals or not vals[geno]:
            print(f"\n{geno.upper()}: (no data)")
            continue

        nb_list = [np.asarray(vals[geno][cond]["avg_nb_volume"], dtype=float) for cond in vals[geno]]
        count_list = [np.asarray(vals[geno][cond]["cell_counts_total"], dtype=float) for cond in vals[geno]]
        lineage_list = [np.asarray(vals[geno][cond]["avg_lineage_volume"], dtype=float) for cond in vals[geno]]

        nb_arr = np.concatenate(nb_list) if nb_list else np.array([], dtype=float)
        count_arr = np.concatenate(count_list) if count_list else np.array([], dtype=float)
        lineage_arr = np.concatenate(lineage_list) if lineage_list else np.array([], dtype=float)

        if nb_arr.size == 0 or count_arr.size == 0 or lineage_arr.size == 0:
            print(f"\n{geno.upper()}: (no data)")
            continue

        print(f"\n{geno.upper()} simulation group:")
        print(
            "  average neuroblast volume "
            f"(voxels / dpn cell): mean={nb_arr.mean():.3f} std={nb_arr.std():.3f} N={nb_arr.size:d}"
        )
        print(
            "  average cell count "
            f"(cells / lineage): mean={count_arr.mean():.3f} std={count_arr.std():.3f} N={count_arr.size:d}"
        )
        print(
            "  average lineage volume "
            f"(voxels / lineage): mean={lineage_arr.mean():.3f} std={lineage_arr.std():.3f} N={lineage_arr.size:d}"
        )
    print("======================================================\n")


def boxplot_metric_for_genotype(
    vals,
    geno,
    metric_key,
    title,
    ylabel,
    reference_condition="01",
):
    dists = []
    labels = []
    sample_sizes = []

    for cond in SIM_CONDITIONS:
        if cond not in vals[geno]:
            continue
        arr = np.asarray(vals[geno][cond][metric_key], dtype=float)
        if arr.size == 0:
            continue
        dists.append(arr)
        labels.append(cond)
        sample_sizes.append(arr.size)

    if not dists:
        print(f"[plot] No data for {geno} / {metric_key}")
        return

    # ---------- statistics ----------
    kw_p = kruskal(*dists).pvalue if len(dists) > 1 else np.nan

    if reference_condition in labels:
        ref_idx = labels.index(reference_condition)
        ref_data = dists[ref_idx]
        ref_n = ref_data.size

        raw_pvals = []
        effect_sizes = []
        for i, arr in enumerate(dists):
            if i == ref_idx:
                raw_pvals.append(np.nan)
                effect_sizes.append(np.nan)
            else:
                u_stat, p = mannwhitneyu(ref_data, arr, alternative="two-sided")
                raw_pvals.append(p)
                effect_sizes.append(rank_biserial_from_u(u_stat, ref_n, arr.size))

        corr_pvals = bonferroni_correct(
            [p for p in raw_pvals if not np.isnan(p)]
        )

        # reinsert NaNs
        corrected = []
        j = 0
        for p in raw_pvals:
            if np.isnan(p):
                corrected.append(np.nan)
            else:
                corrected.append(corr_pvals[j])
                j += 1
    else:
        corrected = [np.nan] * len(dists)
        effect_sizes = [np.nan] * len(dists)

    # ---------- plotting ----------
    display_labels = [f"{cond}\n(n={n})" for cond, n in zip(labels, sample_sizes)]
    means = [float(np.mean(d)) for d in dists]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        dists,
        tick_labels=display_labels,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "tab:blue", "markeredgecolor": "tab:blue", "markersize": 5},
    )
    ax.set_title(
        f"{geno.upper()} simulations — {title}\n"
        f"Kruskal–Wallis p = {kw_p:.2e}"
    )
    ax.set_xlabel("Simulation condition (model_run)")
    ax.xaxis.set_label_coords(0.5, -0.18)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    # ---------- annotate significance ----------
    ymax = max(np.max(d) for d in dists)
    y_range = max(ymax, 1.0)
    y_offset = y_range * 0.05

    # Place mean labels below the axis to avoid overlapping plot elements.
    for i, mean_val in enumerate(means):
        ax.text(
            i + 1,
            -0.10,
            f"μ={mean_val:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color="black",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.2"},
        )

    for i, p in enumerate(corrected):
        star = significance_star(p) if not np.isnan(p) else ""
        if star:
            ax.text(
                i + 1,
                ymax + y_offset,
                star,
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )
            ax.text(
                i + 1,
                ymax + (0.72 * y_offset),
                f"p={p:.2e}",
                ha="center",
                va="top",
                fontsize=8,
                color="black",
            )
            if not np.isnan(effect_sizes[i]):
                ax.text(
                    i + 1,
                    -0.15,
                    f"r_rb={effect_sizes[i]:.2f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                )

    ax.set_ylim(0, ymax + (0.18 * y_range))

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


def boxplot_metric_vs_experimental(
    vals,
    exp_metrics,
    geno,
    metric_key,
    title,
    ylabel,
):
    dists = []
    labels = []
    sample_sizes = []

    for cond in SIM_CONDITIONS:
        if cond not in vals[geno]:
            continue
        arr = np.asarray(vals[geno][cond][metric_key], dtype=float)
        if arr.size == 0:
            continue
        dists.append(arr)
        labels.append(cond)
        sample_sizes.append(arr.size)

    exp_arr = np.asarray(exp_metrics[geno][metric_key], dtype=float)

    if not dists:
        print(f"[exp-compare] No simulation data for {geno} / {metric_key}")
        return
    if exp_arr.size == 0:
        print(f"[exp-compare] No experimental data for {geno} / {metric_key}")
        return

    exp_mean = float(np.mean(exp_arr))
    exp_std = float(np.std(exp_arr))

    raw_pvals = []
    effect_sizes = []
    for arr in dists:
        if arr.size == 0 or exp_arr.size == 0:
            raw_pvals.append(np.nan)
            effect_sizes.append(np.nan)
            continue
        u_stat, p = mannwhitneyu(arr, exp_arr, alternative="two-sided")
        raw_pvals.append(float(p))
        effect_sizes.append(rank_biserial_from_u(u_stat, arr.size, exp_arr.size))

    corr_pvals = bonferroni_correct([p for p in raw_pvals if not np.isnan(p)])

    corrected = []
    j = 0
    for p in raw_pvals:
        if np.isnan(p):
            corrected.append(np.nan)
        else:
            corrected.append(corr_pvals[j])
            j += 1

    display_labels = [f"{cond}\n(n={n})" for cond, n in zip(labels, sample_sizes)]
    means = [float(np.mean(d)) for d in dists]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(
        dists,
        tick_labels=display_labels,
        showfliers=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "tab:blue", "markeredgecolor": "tab:blue", "markersize": 5},
    )

    ax.set_title(
        f"{geno.upper()} simulations vs {geno.upper()} experimental — {title}\n"
        f"Mann–Whitney U per condition vs experimental (Bonferroni corrected)"
    )
    ax.set_xlabel("Simulation condition (model_run)")
    ax.xaxis.set_label_coords(0.5, -0.20)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)

    ymax = max(max(np.max(d) for d in dists), float(np.max(exp_arr)))
    y_range = max(ymax, 1.0)
    y_offset = y_range * 0.05

    ax.axhspan(
        exp_mean - exp_std,
        exp_mean + exp_std,
        color="crimson",
        alpha=0.12,
        zorder=0,
        label=f"Experimental mean ± 1 SD = {exp_mean:.2f} ± {exp_std:.2f}",
    )

    ax.axhline(
        exp_mean,
        color="crimson",
        linestyle="--",
        linewidth=2.0,
        label=f"Experimental mean ({geno.upper()}, n={exp_arr.size}) = {exp_mean:.2f}",
    )

    for i, mean_val in enumerate(means):
        ax.text(
            i + 1,
            -0.10,
            f"mean={mean_val:.2f}",
            ha="center",
            va="top",
            fontsize=9,
            color="black",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "0.8", "boxstyle": "round,pad=0.2"},
        )

    for i, p in enumerate(corrected):
        star = significance_star(p) if not np.isnan(p) else ""
        if star:
            ax.text(
                i + 1,
                ymax + y_offset,
                star,
                ha="center",
                va="bottom",
                fontsize=12,
                color="black",
            )
            ax.text(
                i + 1,
                ymax + (0.72 * y_offset),
                f"p={p:.2e}",
                ha="center",
                va="top",
                fontsize=8,
                color="black",
            )
            if not np.isnan(effect_sizes[i]):
                ax.text(
                    i + 1,
                    -0.17,
                    f"r_rb={effect_sizes[i]:.2f}",
                    ha="center",
                    va="top",
                    fontsize=8,
                    color="black",
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                )

    ax.set_ylim(0, ymax + (0.18 * y_range))
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    geo, counts, model_run, condition = load_sim_npz(SIM_NPZ)
    vals = values_by_genotype_and_condition(geo, counts, model_run, condition)
    exp_metrics = experimental_metrics_by_genotype()

    print_experimental_summary_table(exp_metrics)
    print_simulation_summary_table(vals)
    print_summary_table(vals)

    for geno in GENOTYPES:
        for metric_key, (title, ylabel) in METRICS.items():
            boxplot_metric_for_genotype(vals, geno, metric_key, title, ylabel)

    print("\n=== Comparing simulations to genotype-matched experimental data ===")
    for geno in GENOTYPES:
        for metric_key in EXP_COMPARE_METRICS:
            title, ylabel = METRICS[metric_key]
            boxplot_metric_vs_experimental(
                vals=vals,
                exp_metrics=exp_metrics,
                geno=geno,
                metric_key=metric_key,
                title=title,
                ylabel=ylabel,
            )