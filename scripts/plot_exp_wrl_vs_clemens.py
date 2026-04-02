from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.clemens_data_helpers import (
    build_cell_count_lookup,
    load_clemens_records,
    load_clemens_volume_records,
    maybe_float,
)


@dataclass(frozen=True)
class DatasetMetricConfig:
    key: str
    title: str
    xlabel: str


@dataclass(frozen=True)
class LegacyPlotConfig:
    filename: str
    title: str
    xlabel: str
    mean_digits: int


METRICS = (
    DatasetMetricConfig(
        key="total_cell_count",
        title="Total labeled cells per lineage",
        xlabel="Cells per lineage",
    ),
    DatasetMetricConfig(
        key="lineage_volume_um3",
        title="Lineage volume per lineage",
        xlabel="Lineage volume ($\\mu m^3$)",
    ),
    DatasetMetricConfig(
        key="avg_neuroblast_volume_um3",
        title="Average neuroblast volume per lineage",
        xlabel="Average neuroblast volume ($\\mu m^3$)",
    ),
)

WT_LABEL = "WT"
MUD_LABEL = "mudmut"
WT_COLOR = "tab:blue"
MUD_COLOR = "tab:orange"

LEGACY_PLOTS = (
    LegacyPlotConfig(
        filename="hist_dpn_counts_wt_vs_mud.png",
        title="Neuroblast counts per lineage",
        xlabel="Number of neuroblasts per lineage",
        mean_digits=0,
    ),
    LegacyPlotConfig(
        filename="hist_pros_counts_wt_vs_mud.png",
        title="Non-neuroblast (Pros+) counts per lineage",
        xlabel="Non-neuroblast cells per lineage (Pros)",
        mean_digits=0,
    ),
    LegacyPlotConfig(
        filename="hist_total_labeled_cells_wt_vs_mud.png",
        title="Distribution of labeled cell counts per lineage (WT vs mudmut)",
        xlabel="Number of labeled cells per lineage (Dpn + Pros)",
        mean_digits=0,
    ),
    LegacyPlotConfig(
        filename="hist_lineage_volume_wt_vs_mud.png",
        title="Lineage volumes",
        xlabel="Lineage volume ($\\mu m^3$)",
        mean_digits=1,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot WT vs mudmut histograms for Clemens lineage-level data and "
            "compare the resulting distributions against the WRL dataset."
        )
    )
    parser.add_argument(
        "--clemens-dir",
        type=Path,
        default=Path("data/exp/from_clemens"),
        help="Directory containing the Clemens Excel workbooks.",
    )
    parser.add_argument(
        "--wrl-per-lineage-csv",
        type=Path,
        default=Path("data/exp/wrl_metrics/experimental_wrl_metrics_per_lineage.csv"),
        help="WRL per-lineage summary CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/exp/from_clemens/plots"),
        help="Directory where figures and summary tables will be written.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="auto",
        help="Histogram bins, either an integer or a NumPy binning rule.",
    )
    parser.add_argument(
        "--single-dpn-only",
        action="store_true",
        help=(
            "Optional extra subset after baseline WRL filtering: restrict both "
            "datasets to lineages with exactly one Dpn+ cell."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them.",
    )
    parser.add_argument(
        "--legacy-exp-plot-dir",
        type=Path,
        default=None,
        help=(
            "If set, also write the four legacy Clemens-only histograms with the "
            "same filenames used in data/exp/plots."
        ),
    )
    return parser.parse_args()




def load_wrl_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))

    rows: list[dict[str, object]] = []
    for row in raw_rows:
        rows.append(
            {
                "dataset": "WRL",
                "condition": row["condition"],
                "total_cell_count": maybe_float(row["total_cell_count"]),
                "lineage_volume_um3": maybe_float(row["lineage_volume_um3"]),
                "avg_neuroblast_volume_um3": maybe_float(
                    row["avg_neuroblast_volume_um3"]
                ),
                "n_dpn": int(float(row["n_dpn"])),
            }
        )
    return rows


def print_dataset_counts(
    dataset_name: str,
    rows: list[dict[str, object]],
) -> None:
    wt_n = sum(1 for row in rows if str(row["condition"]) == WT_LABEL)
    mud_n = sum(1 for row in rows if str(row["condition"]) == MUD_LABEL)
    print(f"  {dataset_name}: WT N={wt_n}, mudmut N={mud_n}")


def load_clemens_rows(clemens_dir: Path) -> list[dict[str, object]]:
    clemens_records = load_clemens_records(clemens_dir)
    cell_count_lookup = build_cell_count_lookup(clemens_records)
    volume_records = load_clemens_volume_records(clemens_dir, cell_count_lookup)

    rows: list[dict[str, object]] = []
    for record in volume_records:
        rows.append(
            {
                "dataset": "Clemens",
                "condition": record.condition,
                "total_cell_count": float(record.total_cell_count),
                "lineage_volume_um3": float(record.lineage_volume_um3),
                "avg_neuroblast_volume_um3": float(record.avg_neuroblast_volume_um3),
                "n_dpn": int(record.n_dpn),
            }
        )
    return rows


def filter_rows(
    rows: Iterable[dict[str, object]],
    *,
    single_dpn_only: bool,
) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for row in rows:
        condition = str(row["condition"])
        if condition not in {WT_LABEL, MUD_LABEL}:
            continue
        if single_dpn_only and int(row["n_dpn"]) != 1:
            continue
        filtered.append(row)
    return filtered


def values_for_condition(
    rows: Iterable[dict[str, object]],
    condition: str,
    metric_key: str,
) -> np.ndarray:
    values = [
        maybe_float(row[metric_key])
        for row in rows
        if str(row["condition"]) == condition
        and np.isfinite(maybe_float(row[metric_key]))
    ]
    return np.asarray(values, dtype=float)


def welch_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False, nan_policy="omit").pvalue)


def mann_whitney_pvalue(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    result = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def cliffs_delta(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    greater = 0
    less = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / float(xs.size * ys.size)


def hedges_g(xs: np.ndarray, ys: np.ndarray) -> float:
    n1 = xs.size
    n2 = ys.size
    if n1 < 2 or n2 < 2:
        return float("nan")

    s1 = float(np.std(xs, ddof=1))
    s2 = float(np.std(ys, ddof=1))
    pooled_var = (((n1 - 1) * (s1**2)) + ((n2 - 1) * (s2**2))) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return 0.0

    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    d = (float(np.mean(xs)) - float(np.mean(ys))) / math.sqrt(pooled_var)
    return correction * d


def metric_stats(
    dataset: str, metric_key: str, wt_vals: np.ndarray, mud_vals: np.ndarray
) -> dict[str, object]:
    mw_u, mw_p = mann_whitney_pvalue(wt_vals, mud_vals)
    return {
        "dataset": dataset,
        "metric": metric_key,
        "wt_n": int(wt_vals.size),
        "mud_n": int(mud_vals.size),
        "wt_mean": float(np.mean(wt_vals)) if wt_vals.size else float("nan"),
        "mud_mean": float(np.mean(mud_vals)) if mud_vals.size else float("nan"),
        "wt_std": float(np.std(wt_vals, ddof=1)) if wt_vals.size >= 2 else float("nan"),
        "mud_std": (
            float(np.std(mud_vals, ddof=1)) if mud_vals.size >= 2 else float("nan")
        ),
        "mean_diff_mud_minus_wt": (
            float(np.mean(mud_vals) - np.mean(wt_vals))
            if wt_vals.size and mud_vals.size
            else float("nan")
        ),
        "median_diff_mud_minus_wt": (
            float(np.median(mud_vals) - np.median(wt_vals))
            if wt_vals.size and mud_vals.size
            else float("nan")
        ),
        "welch_pvalue": welch_pvalue(wt_vals, mud_vals),
        "mannwhitney_u": mw_u,
        "mannwhitney_pvalue": mw_p,
        "cliffs_delta_mud_vs_wt": cliffs_delta(mud_vals, wt_vals),
        "hedges_g_mud_vs_wt": hedges_g(mud_vals, wt_vals),
    }


def format_stat(value: float, *, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    if abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def format_mean_label(value: float, digits: int) -> str:
    if not np.isfinite(value):
        return "nan"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def plot_metric_comparison(
    metric: DatasetMetricConfig,
    dataset_rows: dict[str, list[dict[str, object]]],
    stats_rows: list[dict[str, object]],
    outpath: Path,
    bins: str | int,
    show: bool,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8.8), sharey=True)

    for ax, dataset_name in zip(axes, ("WRL", "Clemens")):
        rows = dataset_rows[dataset_name]
        wt_vals = values_for_condition(rows, WT_LABEL, metric.key)
        mud_vals = values_for_condition(rows, MUD_LABEL, metric.key)
        nonempty = [arr for arr in (wt_vals, mud_vals) if arr.size]
        if not nonempty:
            ax.set_visible(False)
            continue

        all_vals = np.concatenate(nonempty)
        bin_edges = np.histogram_bin_edges(all_vals, bins=bins)

        ax.hist(
            wt_vals,
            bins=bin_edges,
            alpha=0.35,
            color=WT_COLOR,
            edgecolor=(0, 0, 0, 0.25),
            linewidth=0.8,
            label=f"WT (N={wt_vals.size})",
        )
        ax.hist(
            mud_vals,
            bins=bin_edges,
            alpha=0.35,
            color=MUD_COLOR,
            edgecolor=(0, 0, 0, 0.25),
            linewidth=0.8,
            label=f"mudmut (N={mud_vals.size})",
        )

        if wt_vals.size:
            ax.axvline(float(np.mean(wt_vals)), color=WT_COLOR, linewidth=2.5)
        if mud_vals.size:
            ax.axvline(float(np.mean(mud_vals)), color=MUD_COLOR, linewidth=2.5)

        stat_row = next(
            row
            for row in stats_rows
            if row["dataset"] == dataset_name and row["metric"] == metric.key
        )
        text = "\n".join(
            [
                f"Welch p={format_stat(float(stat_row['welch_pvalue']))}",
                f"MW p={format_stat(float(stat_row['mannwhitney_pvalue']))}",
                f"Cliff's delta={format_stat(float(stat_row['cliffs_delta_mud_vs_wt']))}",
                f"Hedges g={format_stat(float(stat_row['hedges_g_mud_vs_wt']))}",
            ]
        )
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=20,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "0.8",
            },
        )

        ax.set_title(dataset_name, pad=26)
        ax.set_xlabel(metric.xlabel)
        ax.set_ylabel("Count")
        ax.legend(frameon=False)

    fig.suptitle(metric.title, y=0.992)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.91))
    fig.savefig(outpath, dpi=250, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)


def legacy_metric_values(
    rows: Iterable[dict[str, object]], filename: str, condition: str
) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        if str(row["condition"]) != condition:
            continue
        total_cell_count = maybe_float(row["total_cell_count"])
        lineage_volume_um3 = maybe_float(row["lineage_volume_um3"])
        n_dpn = int(row["n_dpn"])

        if filename == "hist_dpn_counts_wt_vs_mud.png":
            values.append(float(n_dpn))
        elif filename == "hist_pros_counts_wt_vs_mud.png":
            values.append(total_cell_count - n_dpn)
        elif filename == "hist_total_labeled_cells_wt_vs_mud.png":
            values.append(total_cell_count)
        elif filename == "hist_lineage_volume_wt_vs_mud.png":
            # Match the legacy plot scale, which displays these values in units of 1e3 um^3.
            values.append(lineage_volume_um3 / 1000.0)
        else:
            raise ValueError(f"Unsupported legacy plot filename: {filename}")

    return np.asarray([value for value in values if np.isfinite(value)], dtype=float)


def plot_legacy_exp_histogram(
    config: LegacyPlotConfig,
    clemens_rows: list[dict[str, object]],
    outpath: Path,
    show: bool,
) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    wt_vals = legacy_metric_values(clemens_rows, config.filename, WT_LABEL)
    mud_vals = legacy_metric_values(clemens_rows, config.filename, MUD_LABEL)

    fig, ax = plt.subplots(figsize=(14.5, 8.6))
    all_vals = np.concatenate([arr for arr in (wt_vals, mud_vals) if arr.size])
    if config.filename == "hist_dpn_counts_wt_vs_mud.png":
        min_edge = int(np.floor(all_vals.min() - 0.5))
        max_edge = int(np.ceil(all_vals.max() + 0.5))
        bin_edges = np.arange(min_edge, max_edge + 1, 1)
    else:
        bin_edges = np.histogram_bin_edges(all_vals, bins=10)

    wt_mean = float(np.mean(wt_vals)) if wt_vals.size else float("nan")
    mud_mean = float(np.mean(mud_vals)) if mud_vals.size else float("nan")
    welch_p = welch_pvalue(wt_vals, mud_vals)

    ax.hist(
        wt_vals,
        bins=bin_edges,
        alpha=0.4,
        color=WT_COLOR,
        edgecolor="black",
        linewidth=1.0,
        label=f"WT (N = {wt_vals.size})",
    )
    ax.hist(
        mud_vals,
        bins=bin_edges,
        alpha=0.4,
        color=MUD_COLOR,
        edgecolor="black",
        linewidth=1.0,
        label=f"Mudmut (N = {mud_vals.size})",
    )
    ax.axvline(
        wt_mean,
        color=WT_COLOR,
        linewidth=2.5,
        label=f"WT mean = {format_mean_label(wt_mean, config.mean_digits)}",
    )
    ax.axvline(
        mud_mean,
        color=MUD_COLOR,
        linewidth=2.5,
        label=f"mudmut mean = {format_mean_label(mud_mean, config.mean_digits)}",
    )
    ax.plot([], [], " ", label=f"Welch's t-test: p = {welch_p:.2e}")

    ax.set_title(config.title, pad=28)
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel("Count")
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.992),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.96))
    fig.savefig(outpath, dpi=250, bbox_inches="tight", pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)


def write_stats_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "metric",
        "wt_n",
        "mud_n",
        "wt_mean",
        "mud_mean",
        "wt_std",
        "mud_std",
        "mean_diff_mud_minus_wt",
        "median_diff_mud_minus_wt",
        "welch_pvalue",
        "mannwhitney_u",
        "mannwhitney_pvalue",
        "cliffs_delta_mud_vs_wt",
        "hedges_g_mud_vs_wt",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": 26,
            "axes.titlesize": 34,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 24,
        }
    )

    try:
        bins: str | int = int(args.bins)
    except ValueError:
        bins = args.bins

    dataset_rows = {
        "WRL": filter_rows(
            load_wrl_rows(args.wrl_per_lineage_csv),
            single_dpn_only=args.single_dpn_only,
        ),
        "Clemens": filter_rows(
            load_clemens_rows(args.clemens_dir), single_dpn_only=args.single_dpn_only
        ),
    }

    print("Lineage counts used for plotting:")
    print_dataset_counts("WRL", dataset_rows["WRL"])
    print_dataset_counts("Clemens", dataset_rows["Clemens"])

    stats_rows: list[dict[str, object]] = []
    for dataset_name, rows in dataset_rows.items():
        for metric in METRICS:
            wt_vals = values_for_condition(rows, WT_LABEL, metric.key)
            mud_vals = values_for_condition(rows, MUD_LABEL, metric.key)
            stats_rows.append(metric_stats(dataset_name, metric.key, wt_vals, mud_vals))

    stats_csv = args.outdir / "wt_vs_mud_stats_wrl_vs_clemens.csv"
    write_stats_csv(stats_csv, stats_rows)

    for metric in METRICS:
        outpath = args.outdir / f"{metric.key}_wt_vs_mud_wrl_vs_clemens.png"
        plot_metric_comparison(
            metric=metric,
            dataset_rows=dataset_rows,
            stats_rows=stats_rows,
            outpath=outpath,
            bins=bins,
            show=args.show,
        )

    if args.legacy_exp_plot_dir is not None:
        clemens_rows = dataset_rows["Clemens"]
        for config in LEGACY_PLOTS:
            plot_legacy_exp_histogram(
                config=config,
                clemens_rows=clemens_rows,
                outpath=args.legacy_exp_plot_dir / config.filename,
                show=args.show,
            )

    print("WT vs mudmut comparison:")
    for row in stats_rows:
        print(
            f"  {row['dataset']} | {row['metric']}:"
            f" WT mean={format_stat(float(row['wt_mean']))}"
            f" | mud mean={format_stat(float(row['mud_mean']))}"
            f" | diff(mud-WT)={format_stat(float(row['mean_diff_mud_minus_wt']))}"
            f" | Welch p={format_stat(float(row['welch_pvalue']))}"
            f" | MW p={format_stat(float(row['mannwhitney_pvalue']))}"
            f" | Cliff's delta={format_stat(float(row['cliffs_delta_mud_vs_wt']))}"
            f" | Hedges g={format_stat(float(row['hedges_g_mud_vs_wt']))}"
        )

    print("\nSaved:")
    print(f"  {stats_csv}")
    for metric in METRICS:
        print(f"  {args.outdir / f'{metric.key}_wt_vs_mud_wrl_vs_clemens.png'}")
    if args.legacy_exp_plot_dir is not None:
        for config in LEGACY_PLOTS:
            print(f"  {args.legacy_exp_plot_dir / config.filename}")


if __name__ == "__main__":
    main()
