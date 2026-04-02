from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal

from src.npz_metrics import compute_metrics

# -------------------------------
# Config
# -------------------------------
SIM_NPZ = Path("data/sim/sim_geo_counts_endpoints.npz")
SIM_CONDITIONS = [f"sim{i:02d}" for i in range(1, 41)]
DEFAULT_PROJECTED_DS_UM = 0.3
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
    "lineage_volumes": (
        "Lineage projected areas per lineage",
        "um^2 (dpn ∪ pros-only)",
    ),
    "avg_nb_volume": (
        "Average neuroblast projected area per lineage",
        "um^2 / dpn_cell_count",
    ),
    "avg_lineage_volume": (
        "Average lineage projected area per lineage",
        "um^2 (dpn ∪ pros-only)",
    ),
}

EXP_COMPARE_METRICS = ["cell_counts_total", "lineage_volumes", "avg_nb_volume"]
AREA_LIKE_METRICS = {"lineage_volumes", "avg_nb_volume", "avg_lineage_volume"}

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


def normalize_ds_value(ds_value) -> float | None:
    if ds_value is None:
        return None
    arr = np.asarray(ds_value, dtype=float)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def convert_metric_values(
    values: np.ndarray, metric_key: str, ds_um: float
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if metric_key in AREA_LIKE_METRICS:
        return arr * (ds_um**2)
    return arr



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
        raise KeyError(
            "NPZ is missing 'condition'. Re-save sim NPZ with condition included."
        )
    ds_um = normalize_ds_value(data.get("ds", None))
    return geo, counts, model_run, condition, ds_um


def load_experimental_arrays(
    npz_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, float | None]:
    if not npz_dir.exists():
        raise FileNotFoundError(f"Experimental NPZ directory not found: {npz_dir}")

    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No .npz files found in experimental directory: {npz_dir}"
        )

    geo_list: List[np.ndarray] = []
    counts_list: List[np.ndarray] = []
    ds_values: List[float] = []

    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        if "geo" not in data or "counts" not in data:
            raise KeyError(f"{fpath} is missing required keys 'geo' and/or 'counts'.")
        geo_list.append(data["geo"])
        counts_list.append(data["counts"])
        ds_value = normalize_ds_value(data.get("ds", None))
        if ds_value is not None:
            ds_values.append(ds_value)

    geo = np.concatenate(geo_list, axis=0)
    counts = np.concatenate(counts_list, axis=0)
    ds_um = None
    if ds_values:
        ds_arr = np.asarray(ds_values, dtype=float)
        if not np.allclose(ds_arr, ds_arr[0]):
            raise ValueError(f"Inconsistent ds values found in {npz_dir}: {ds_values}")
        ds_um = float(ds_arr[0])
    return geo, counts, ds_um


def experimental_metrics_by_genotype() -> (
    Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, float | None]]
):
    out: Dict[str, Dict[str, np.ndarray]] = {}
    ds_by_genotype: Dict[str, float | None] = {}
    for geno, exp_dir in EXP_NPZ_DIRS.items():
        geo, counts, ds_um = load_experimental_arrays(exp_dir)
        out[geno] = compute_metrics(geo, counts)
        ds_by_genotype[geno] = ds_um
    return out, ds_by_genotype


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
def print_summary_table(
    vals: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ds_um: float,
):
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
            for key in [
                "cell_counts_total",
                "lineage_volumes",
                "avg_nb_volume",
                "avg_lineage_volume",
            ]:
                arr = convert_metric_values(vals[geno][cond][key], key, ds_um)
                if arr.size == 0:
                    continue
                print(
                    f"    {key:18s}  mean={arr.mean():.3f}  std={arr.std():.3f}  N={arr.size:d}"
                )
    print("========================================================\n")


def print_experimental_summary_table(
    exp_metrics: Dict[str, Dict[str, np.ndarray]],
    ds_by_genotype: Dict[str, float | None],
):
    print("\n=== Experimental Summary (aggregate over all lineages) ===")
    for geno in EXP_NPZ_DIRS:
        if geno not in exp_metrics:
            print(f"\n{geno.upper()}: (no data)")
            continue

        ds_um = ds_by_genotype.get(geno) or DEFAULT_PROJECTED_DS_UM
        nb_arr = convert_metric_values(
            exp_metrics[geno]["avg_nb_volume"], "avg_nb_volume", ds_um
        )
        count_arr = np.asarray(exp_metrics[geno]["cell_counts_total"], dtype=float)
        lineage_arr = convert_metric_values(
            exp_metrics[geno]["avg_lineage_volume"], "avg_lineage_volume", ds_um
        )

        print(f"\n{geno.upper()} experimental group:")
        print(
            "  average neuroblast projected area "
            f"(um^2 / dpn cell): mean={nb_arr.mean():.3f} std={nb_arr.std():.3f} N={nb_arr.size:d}"
        )
        print(
            "  average cell count "
            f"(cells / lineage): mean={count_arr.mean():.3f} std={count_arr.std():.3f} N={count_arr.size:d}"
        )
        print(
            "  average lineage projected area "
            f"(um^2 / lineage): mean={lineage_arr.mean():.3f} std={lineage_arr.std():.3f} N={lineage_arr.size:d}"
        )
    print("=======================================================\n")


def print_simulation_summary_table(
    vals: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    ds_um: float,
):
    print("\n=== Simulation Summary (aggregate over all lineages) ===")
    for geno in EXP_NPZ_DIRS:
        if geno not in vals or not vals[geno]:
            print(f"\n{geno.upper()}: (no data)")
            continue

        nb_list = [
            convert_metric_values(
                vals[geno][cond]["avg_nb_volume"], "avg_nb_volume", ds_um
            )
            for cond in vals[geno]
        ]
        count_list = [
            np.asarray(vals[geno][cond]["cell_counts_total"], dtype=float)
            for cond in vals[geno]
        ]
        lineage_list = [
            convert_metric_values(
                vals[geno][cond]["avg_lineage_volume"], "avg_lineage_volume", ds_um
            )
            for cond in vals[geno]
        ]

        nb_arr = np.concatenate(nb_list) if nb_list else np.array([], dtype=float)
        count_arr = (
            np.concatenate(count_list) if count_list else np.array([], dtype=float)
        )
        lineage_arr = (
            np.concatenate(lineage_list) if lineage_list else np.array([], dtype=float)
        )

        if nb_arr.size == 0 or count_arr.size == 0 or lineage_arr.size == 0:
            print(f"\n{geno.upper()}: (no data)")
            continue

        print(f"\n{geno.upper()} simulation group:")
        print(
            "  average neuroblast projected area "
            f"(um^2 / dpn cell): mean={nb_arr.mean():.3f} std={nb_arr.std():.3f} N={nb_arr.size:d}"
        )
        print(
            "  average cell count "
            f"(cells / lineage): mean={count_arr.mean():.3f} std={count_arr.std():.3f} N={count_arr.size:d}"
        )
        print(
            "  average lineage projected area "
            f"(um^2 / lineage): mean={lineage_arr.mean():.3f} std={lineage_arr.std():.3f} N={lineage_arr.size:d}"
        )
    print("======================================================\n")


def boxplot_metric_for_genotype(
    vals,
    geno,
    metric_key,
    title,
    ylabel,
    ds_um,
    reference_condition="sim01",
):
    dists = []
    labels = []
    sample_sizes = []

    for cond in SIM_CONDITIONS:
        if cond not in vals[geno]:
            continue
        arr = convert_metric_values(vals[geno][cond][metric_key], metric_key, ds_um)
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

        corr_pvals = bonferroni_correct([p for p in raw_pvals if not np.isnan(p)])

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
        meanprops={
            "marker": "D",
            "markerfacecolor": "tab:blue",
            "markeredgecolor": "tab:blue",
            "markersize": 5,
        },
    )
    ax.set_title(
        f"{geno.upper()} simulations — {title}\n" f"Kruskal–Wallis p = {kw_p:.2e}"
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
            bbox={
                "facecolor": "white",
                "edgecolor": "0.8",
                "boxstyle": "round,pad=0.2",
            },
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
    sim_ds_um,
    exp_ds_um,
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
        arr = convert_metric_values(vals[geno][cond][metric_key], metric_key, sim_ds_um)
        if arr.size == 0:
            continue
        dists.append(arr)
        labels.append(cond)
        sample_sizes.append(arr.size)

    exp_arr = convert_metric_values(
        exp_metrics[geno][metric_key], metric_key, exp_ds_um
    )

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
        meanprops={
            "marker": "D",
            "markerfacecolor": "tab:blue",
            "markeredgecolor": "tab:blue",
            "markersize": 5,
        },
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
            bbox={
                "facecolor": "white",
                "edgecolor": "0.8",
                "boxstyle": "round,pad=0.2",
            },
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
    geo, counts, model_run, condition, sim_ds_um = load_sim_npz(SIM_NPZ)
    vals = values_by_genotype_and_condition(geo, counts, model_run, condition)
    exp_metrics, exp_ds_by_genotype = experimental_metrics_by_genotype()
    resolved_sim_ds_um = sim_ds_um
    if resolved_sim_ds_um is None:
        exp_ds_values = [ds for ds in exp_ds_by_genotype.values() if ds is not None]
        resolved_sim_ds_um = (
            exp_ds_values[0] if exp_ds_values else DEFAULT_PROJECTED_DS_UM
        )
        print(
            "[info] Simulation NPZ is missing 'ds'; "
            f"using ds={resolved_sim_ds_um:.3f} um for projected-area conversion."
        )

    print_experimental_summary_table(exp_metrics, exp_ds_by_genotype)
    print_simulation_summary_table(vals, resolved_sim_ds_um)
    print_summary_table(vals, resolved_sim_ds_um)

    for geno in GENOTYPES:
        for metric_key, (title, ylabel) in METRICS.items():
            boxplot_metric_for_genotype(
                vals,
                geno,
                metric_key,
                title,
                ylabel,
                resolved_sim_ds_um,
            )

    print("\n=== Comparing simulations to genotype-matched experimental data ===")
    for geno in GENOTYPES:
        for metric_key in EXP_COMPARE_METRICS:
            title, ylabel = METRICS[metric_key]
            boxplot_metric_vs_experimental(
                vals=vals,
                exp_metrics=exp_metrics,
                sim_ds_um=resolved_sim_ds_um,
                exp_ds_um=exp_ds_by_genotype.get(geno) or DEFAULT_PROJECTED_DS_UM,
                geno=geno,
                metric_key=metric_key,
                title=title,
                ylabel=ylabel,
            )
