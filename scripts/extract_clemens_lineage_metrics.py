from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from src.clemens_data_helpers import (
    ClemensCellCountRecord,
    ClemensVolumeRecord,
    build_cell_count_lookup,
    load_clemens_records,
    load_clemens_volume_records,
    maybe_float,
)
from src.npz_metrics import mean_std_n

SUMMARY_FIELDNAMES = [
    "condition",
    "lineages_total",
    "lineages_kept",
    "lineages_skipped_disconnected",
    "lineages_skipped_no_dpn",
    "mean_lineage_volume_um3",
    "std_lineage_volume_um3",
    "mean_lineage_projected_area_um2",
    "std_lineage_projected_area_um2",
    "mean_avg_neuroblast_volume_um3",
    "std_avg_neuroblast_volume_um3",
    "mean_avg_neuroblast_projected_area_um2",
    "std_avg_neuroblast_projected_area_um2",
    "mean_total_cell_count",
    "std_total_cell_count",
    "n_lineages_in_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Clemens Pros cell-count spreadsheets to the WRL cell-count "
            "summary by treating each nonzero spreadsheet entry as one lineage."
        )
    )
    parser.add_argument(
        "--clemens-dir",
        type=Path,
        default=Path("data/exp/from_clemens"),
        help="Directory containing the Clemens Excel workbooks.",
    )
    parser.add_argument(
        "--wrl-summary-csv",
        type=Path,
        default=Path("data/exp/wrl_metrics/experimental_wrl_metrics_summary.csv"),
        help="Existing WRL summary CSV to compare against.",
    )
    parser.add_argument(
        "--wrl-per-lineage-csv",
        type=Path,
        default=Path("data/exp/wrl_metrics/experimental_wrl_metrics_per_lineage.csv"),
        help="Existing WRL per-lineage CSV used for statistical tests.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/exp/from_clemens"),
        help="Directory for generated Clemens summary and comparison CSVs.",
    )
    parser.add_argument(
        "--single-dpn-only",
        action="store_true",
        help=(
            "Restrict WRL rows and Clemens volume-linked rows to lineages with "
            "exactly one Dpn+ cell."
        ),
    )
    return parser.parse_args()


def summarize_clemens_records(
    records: list[ClemensCellCountRecord],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []

    for condition in ("WT", "mudmut", "nanobody"):
        condition_values = [
            record.total_cell_count for record in records if record.condition == condition
        ]
        mean_count, std_count, n_values = mean_std_n(condition_values)
        summaries.append(
            {
                "condition": condition,
                "lineages_total": n_values,
                "lineages_kept": n_values,
                "lineages_skipped_disconnected": 0,
                "lineages_skipped_no_dpn": 0,
                "mean_lineage_volume_um3": float("nan"),
                "std_lineage_volume_um3": float("nan"),
                "mean_lineage_projected_area_um2": float("nan"),
                "std_lineage_projected_area_um2": float("nan"),
                "mean_avg_neuroblast_volume_um3": float("nan"),
                "std_avg_neuroblast_volume_um3": float("nan"),
                "mean_avg_neuroblast_projected_area_um2": float("nan"),
                "std_avg_neuroblast_projected_area_um2": float("nan"),
                "mean_total_cell_count": mean_count,
                "std_total_cell_count": std_count,
                "n_lineages_in_summary": n_values,
            }
        )

    return summaries


def summarize_clemens_volume_records(
    records: list[ClemensVolumeRecord],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []

    for condition in ("WT", "mudmut", "nanobody"):
        condition_records = [record for record in records if record.condition == condition]
        lineage_volumes = [record.lineage_volume_um3 for record in condition_records]
        lineage_projected_areas = [
            record.lineage_projected_area_um2 for record in condition_records
        ]
        avg_dpn_volumes = [record.avg_neuroblast_volume_um3 for record in condition_records]
        avg_dpn_projected_areas = [
            record.avg_neuroblast_projected_area_um2 for record in condition_records
        ]
        cell_counts = [
            record.total_cell_count
            for record in condition_records
            if not math.isnan(record.total_cell_count)
        ]

        summaries.append(
            {
                "condition": condition,
                "lineages_total": len(condition_records),
                "lineages_kept": len(condition_records),
                "lineages_skipped_disconnected": 0,
                "lineages_skipped_no_dpn": 0,
                "mean_lineage_volume_um3": mean_std_n(lineage_volumes)[0],
                "std_lineage_volume_um3": mean_std_n(lineage_volumes)[1],
                "mean_lineage_projected_area_um2": mean_std_n(lineage_projected_areas)[0],
                "std_lineage_projected_area_um2": mean_std_n(lineage_projected_areas)[1],
                "mean_avg_neuroblast_volume_um3": mean_std_n(avg_dpn_volumes)[0],
                "std_avg_neuroblast_volume_um3": mean_std_n(avg_dpn_volumes)[1],
                "mean_avg_neuroblast_projected_area_um2": mean_std_n(
                    avg_dpn_projected_areas
                )[0],
                "std_avg_neuroblast_projected_area_um2": mean_std_n(
                    avg_dpn_projected_areas
                )[1],
                "mean_total_cell_count": mean_std_n(cell_counts)[0],
                "std_total_cell_count": mean_std_n(cell_counts)[1],
                "n_lineages_in_summary": len(condition_records),
            }
        )

    return summaries


def write_clemens_per_lineage_csv(
    path: Path, records: list[ClemensCellCountRecord]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "lobe", "lineage", "total_cell_count"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_clemens_volume_per_lineage_csv(
    path: Path,
    records: list[ClemensVolumeRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "lobe",
        "lineage",
        "total_cell_count",
        "lineage_volume_um3",
        "lineage_projected_area_um2",
        "avg_neuroblast_volume_um3",
        "avg_neuroblast_projected_area_um2",
        "n_dpn",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_summary_csv(path: Path, summaries: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_wrl_per_lineage_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in load_csv_rows(path):
        rows.append(
            {
                "condition": row["condition"],
                "lobe": row["lobe"],
                "lineage_index": int(float(row["lineage_index"])),
                "n_dpn": int(float(row["n_dpn"])),
                "n_pros": int(float(row["n_pros"])),
                "total_cell_count": maybe_float(row["total_cell_count"]),
                "lineage_volume_um3": maybe_float(row["lineage_volume_um3"]),
                "lineage_projected_area_um2": maybe_float(
                    row["lineage_projected_area_um2"]
                ),
                "avg_neuroblast_volume_um3": maybe_float(
                    row["avg_neuroblast_volume_um3"]
                ),
                "avg_neuroblast_projected_area_um2": maybe_float(
                    row["avg_neuroblast_projected_area_um2"]
                ),
            }
        )
    return rows


def filter_wrl_per_lineage_rows(
    rows: list[dict[str, object]],
    *,
    single_dpn_only: bool,
) -> list[dict[str, object]]:
    if not single_dpn_only:
        return rows
    return [row for row in rows if int(row["n_dpn"]) == 1]


def summarize_wrl_per_lineage_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, float | int | str]]:
    summaries: list[dict[str, float | int | str]] = []
    for condition in ("WT", "mudmut", "nanobody"):
        condition_rows = [row for row in rows if row["condition"] == condition]
        lineage_volumes = [float(row["lineage_volume_um3"]) for row in condition_rows]
        lineage_projected_areas = [
            float(row["lineage_projected_area_um2"]) for row in condition_rows
        ]
        avg_dpn_volumes = [
            float(row["avg_neuroblast_volume_um3"]) for row in condition_rows
        ]
        avg_dpn_projected_areas = [
            float(row["avg_neuroblast_projected_area_um2"]) for row in condition_rows
        ]
        cell_counts = [
            float(row["total_cell_count"])
            for row in condition_rows
            if not math.isnan(float(row["total_cell_count"]))
        ]

        summaries.append(
            {
                "condition": condition,
                "lineages_total": len(condition_rows),
                "lineages_kept": len(condition_rows),
                "lineages_skipped_disconnected": 0,
                "lineages_skipped_no_dpn": 0,
                "mean_lineage_volume_um3": mean_std_n(lineage_volumes)[0],
                "std_lineage_volume_um3": mean_std_n(lineage_volumes)[1],
                "mean_lineage_projected_area_um2": mean_std_n(
                    lineage_projected_areas
                )[0],
                "std_lineage_projected_area_um2": mean_std_n(
                    lineage_projected_areas
                )[1],
                "mean_avg_neuroblast_volume_um3": mean_std_n(avg_dpn_volumes)[0],
                "std_avg_neuroblast_volume_um3": mean_std_n(avg_dpn_volumes)[1],
                "mean_avg_neuroblast_projected_area_um2": mean_std_n(
                    avg_dpn_projected_areas
                )[0],
                "std_avg_neuroblast_projected_area_um2": mean_std_n(
                    avg_dpn_projected_areas
                )[1],
                "mean_total_cell_count": mean_std_n(cell_counts)[0],
                "std_total_cell_count": mean_std_n(cell_counts)[1],
                "n_lineages_in_summary": len(condition_rows),
            }
        )

    return summaries


def load_wrl_cell_count_values_from_rows(
    rows: list[dict[str, object]],
) -> dict[str, list[float]]:
    values_by_condition: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row["condition"])
        total_cell_count = maybe_float(row["total_cell_count"])
        if math.isnan(total_cell_count):
            continue
        values_by_condition[condition].append(total_cell_count)
    return dict(values_by_condition)



def build_cell_count_comparison_rows(
    wrl_rows: list[dict[str, str]],
    clemens_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    wrl_by_condition = {row["condition"]: row for row in wrl_rows}
    comparison_rows: list[dict[str, object]] = []

    for clemens_row in clemens_rows:
        condition = str(clemens_row["condition"])
        wrl_row = wrl_by_condition.get(condition)
        if wrl_row is None:
            continue

        clemens_mean = float(clemens_row["mean_total_cell_count"])
        clemens_std = float(clemens_row["std_total_cell_count"])
        clemens_n = int(clemens_row["n_lineages_in_summary"])
        wrl_mean = maybe_float(wrl_row["mean_total_cell_count"])
        wrl_std = maybe_float(wrl_row["std_total_cell_count"])
        wrl_n = int(float(wrl_row["n_lineages_in_summary"]))

        comparison_rows.append(
            {
                "condition": condition,
                "wrl_mean_total_cell_count": wrl_mean,
                "clemens_mean_total_cell_count": clemens_mean,
                "mean_total_cell_count_diff_clemens_minus_wrl": clemens_mean - wrl_mean,
                "wrl_std_total_cell_count": wrl_std,
                "clemens_std_total_cell_count": clemens_std,
                "std_total_cell_count_diff_clemens_minus_wrl": clemens_std - wrl_std,
                "wrl_n_lineages_in_summary": wrl_n,
                "clemens_n_lineages_in_summary": clemens_n,
                "n_lineages_diff_clemens_minus_wrl": clemens_n - wrl_n,
            }
        )

    return comparison_rows


def build_volume_comparison_rows(
    wrl_rows: list[dict[str, str]],
    clemens_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    wrl_by_condition = {row["condition"]: row for row in wrl_rows}
    comparison_rows: list[dict[str, object]] = []

    metric_names = [
        "mean_lineage_volume_um3",
        "std_lineage_volume_um3",
        "mean_lineage_projected_area_um2",
        "std_lineage_projected_area_um2",
        "mean_avg_neuroblast_volume_um3",
        "std_avg_neuroblast_volume_um3",
        "mean_avg_neuroblast_projected_area_um2",
        "std_avg_neuroblast_projected_area_um2",
        "mean_total_cell_count",
        "std_total_cell_count",
    ]

    for clemens_row in clemens_rows:
        condition = str(clemens_row["condition"])
        wrl_row = wrl_by_condition.get(condition)
        if wrl_row is None:
            continue

        row: dict[str, object] = {
            "condition": condition,
            "wrl_n_lineages_in_summary": int(float(wrl_row["n_lineages_in_summary"])),
            "clemens_n_lineages_in_summary": int(clemens_row["n_lineages_in_summary"]),
            "n_lineages_diff_clemens_minus_wrl": int(clemens_row["n_lineages_in_summary"])
            - int(float(wrl_row["n_lineages_in_summary"])),
        }
        for metric_name in metric_names:
            wrl_value = maybe_float(wrl_row[metric_name])
            clemens_value = float(clemens_row[metric_name])
            row[f"wrl_{metric_name}"] = wrl_value
            row[f"clemens_{metric_name}"] = clemens_value
            row[f"{metric_name}_diff_clemens_minus_wrl"] = clemens_value - wrl_value
        comparison_rows.append(row)

    return comparison_rows


def cliffs_delta(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return float("nan")
    greater = 0
    less = 0
    for x in xs:
        for y in ys:
            if x > y:
                greater += 1
            elif x < y:
                less += 1
    return (greater - less) / (len(xs) * len(ys))


def build_statistical_test_rows(
    wrl_values_by_condition: dict[str, list[float]],
    clemens_records: list[ClemensCellCountRecord],
) -> list[dict[str, object]]:
    clemens_values_by_condition: dict[str, list[float]] = defaultdict(list)
    for record in clemens_records:
        clemens_values_by_condition[record.condition].append(record.total_cell_count)

    rows: list[dict[str, object]] = []
    for condition in ("WT", "mudmut", "nanobody"):
        wrl_values = wrl_values_by_condition.get(condition, [])
        clemens_values = clemens_values_by_condition.get(condition, [])
        if not wrl_values or not clemens_values:
            continue

        welch = stats.ttest_ind(
            wrl_values,
            clemens_values,
            equal_var=False,
            nan_policy="omit",
        )
        mann_whitney = stats.mannwhitneyu(
            wrl_values,
            clemens_values,
            alternative="two-sided",
        )
        ks = stats.ks_2samp(wrl_values, clemens_values, alternative="two-sided")

        rows.append(
            {
                "condition": condition,
                "wrl_n": len(wrl_values),
                "clemens_n": len(clemens_values),
                "wrl_mean_total_cell_count": float(np.mean(wrl_values)),
                "clemens_mean_total_cell_count": float(np.mean(clemens_values)),
                "mean_diff_clemens_minus_wrl": float(np.mean(clemens_values) - np.mean(wrl_values)),
                "welch_t_statistic": float(welch.statistic),
                "welch_t_pvalue": float(welch.pvalue),
                "mannwhitney_u_statistic": float(mann_whitney.statistic),
                "mannwhitney_pvalue": float(mann_whitney.pvalue),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "cliffs_delta_clemens_vs_wrl": float(cliffs_delta(clemens_values, wrl_values)),
            }
        )

    return rows


def write_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "wrl_mean_total_cell_count",
        "clemens_mean_total_cell_count",
        "mean_total_cell_count_diff_clemens_minus_wrl",
        "wrl_std_total_cell_count",
        "clemens_std_total_cell_count",
        "std_total_cell_count_diff_clemens_minus_wrl",
        "wrl_n_lineages_in_summary",
        "clemens_n_lineages_in_summary",
        "n_lineages_diff_clemens_minus_wrl",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_volume_comparison_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "wrl_n_lineages_in_summary",
        "clemens_n_lineages_in_summary",
        "n_lineages_diff_clemens_minus_wrl",
        "wrl_mean_lineage_volume_um3",
        "clemens_mean_lineage_volume_um3",
        "mean_lineage_volume_um3_diff_clemens_minus_wrl",
        "wrl_std_lineage_volume_um3",
        "clemens_std_lineage_volume_um3",
        "std_lineage_volume_um3_diff_clemens_minus_wrl",
        "wrl_mean_lineage_projected_area_um2",
        "clemens_mean_lineage_projected_area_um2",
        "mean_lineage_projected_area_um2_diff_clemens_minus_wrl",
        "wrl_std_lineage_projected_area_um2",
        "clemens_std_lineage_projected_area_um2",
        "std_lineage_projected_area_um2_diff_clemens_minus_wrl",
        "wrl_mean_avg_neuroblast_volume_um3",
        "clemens_mean_avg_neuroblast_volume_um3",
        "mean_avg_neuroblast_volume_um3_diff_clemens_minus_wrl",
        "wrl_std_avg_neuroblast_volume_um3",
        "clemens_std_avg_neuroblast_volume_um3",
        "std_avg_neuroblast_volume_um3_diff_clemens_minus_wrl",
        "wrl_mean_avg_neuroblast_projected_area_um2",
        "clemens_mean_avg_neuroblast_projected_area_um2",
        "mean_avg_neuroblast_projected_area_um2_diff_clemens_minus_wrl",
        "wrl_std_avg_neuroblast_projected_area_um2",
        "clemens_std_avg_neuroblast_projected_area_um2",
        "std_avg_neuroblast_projected_area_um2_diff_clemens_minus_wrl",
        "wrl_mean_total_cell_count",
        "clemens_mean_total_cell_count",
        "mean_total_cell_count_diff_clemens_minus_wrl",
        "wrl_std_total_cell_count",
        "clemens_std_total_cell_count",
        "std_total_cell_count_diff_clemens_minus_wrl",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_statistical_tests_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "wrl_n",
        "clemens_n",
        "wrl_mean_total_cell_count",
        "clemens_mean_total_cell_count",
        "mean_diff_clemens_minus_wrl",
        "welch_t_statistic",
        "welch_t_pvalue",
        "mannwhitney_u_statistic",
        "mannwhitney_pvalue",
        "ks_statistic",
        "ks_pvalue",
        "cliffs_delta_clemens_vs_wrl",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_number(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()

    clemens_records = load_clemens_records(args.clemens_dir)
    cell_count_lookup = build_cell_count_lookup(clemens_records)
    clemens_volume_records = load_clemens_volume_records(
        args.clemens_dir,
        cell_count_lookup,
    )

    if args.single_dpn_only:
        clemens_volume_records = [
            record for record in clemens_volume_records if record.n_dpn == 1
        ]
        clemens_records_for_counts = [
            ClemensCellCountRecord(
                condition=record.condition,
                lobe=record.lobe,
                lineage=record.lineage,
                total_cell_count=record.total_cell_count,
            )
            for record in clemens_volume_records
            if not math.isnan(record.total_cell_count)
        ]
    else:
        clemens_records_for_counts = clemens_records

    clemens_summaries = summarize_clemens_records(clemens_records_for_counts)
    clemens_volume_summaries = summarize_clemens_volume_records(clemens_volume_records)

    wrl_per_lineage_rows = filter_wrl_per_lineage_rows(
        load_wrl_per_lineage_rows(args.wrl_per_lineage_csv),
        single_dpn_only=args.single_dpn_only,
    )
    wrl_rows = summarize_wrl_per_lineage_rows(wrl_per_lineage_rows)
    wrl_values_by_condition = load_wrl_cell_count_values_from_rows(wrl_per_lineage_rows)

    comparison_rows = build_cell_count_comparison_rows(wrl_rows, clemens_summaries)
    volume_comparison_rows = build_volume_comparison_rows(
        wrl_rows,
        clemens_volume_summaries,
    )
    statistical_test_rows = build_statistical_test_rows(
        wrl_values_by_condition,
        clemens_records_for_counts,
    )

    per_lineage_csv = args.outdir / "clemens_cell_count_per_lineage.csv"
    clemens_summary_csv = args.outdir / "clemens_cell_count_summary.csv"
    comparison_csv = args.outdir / "clemens_vs_wrl_cell_count_comparison.csv"
    tests_csv = args.outdir / "clemens_vs_wrl_cell_count_statistical_tests.csv"
    volume_per_lineage_csv = args.outdir / "clemens_volume_per_lineage.csv"
    volume_summary_csv = args.outdir / "clemens_volume_summary.csv"
    volume_comparison_csv = args.outdir / "clemens_vs_wrl_volume_comparison.csv"

    write_clemens_per_lineage_csv(per_lineage_csv, clemens_records_for_counts)
    write_summary_csv(clemens_summary_csv, clemens_summaries)
    write_comparison_csv(comparison_csv, comparison_rows)
    write_statistical_tests_csv(tests_csv, statistical_test_rows)
    write_clemens_volume_per_lineage_csv(
        volume_per_lineage_csv,
        clemens_volume_records,
    )
    write_summary_csv(volume_summary_csv, clemens_volume_summaries)
    write_volume_comparison_csv(volume_comparison_csv, volume_comparison_rows)

    print("Clemens cell-count summary:")
    for row in clemens_summaries:
        print(
            f"  {row['condition']}:"
            f" mean={format_number(float(row['mean_total_cell_count']))}"
            f" std={format_number(float(row['std_total_cell_count']))}"
            f" N={int(row['n_lineages_in_summary'])}"
        )

    print("\nWRL summary used for comparison:")
    for row in wrl_rows:
        print(
            f"  {row['condition']}:"
            f" mean={format_number(float(row['mean_total_cell_count']))}"
            f" std={format_number(float(row['std_total_cell_count']))}"
            f" N={int(row['n_lineages_in_summary'])}"
        )

    print("\nStatistical tests vs WRL:")
    for row in statistical_test_rows:
        print(
            f"  {row['condition']}:"
            f" Welch p={format_number(float(row['welch_t_pvalue']))}"
            f" | Mann-Whitney p={format_number(float(row['mannwhitney_pvalue']))}"
            f" | KS p={format_number(float(row['ks_pvalue']))}"
            f" | Cliff's delta={format_number(float(row['cliffs_delta_clemens_vs_wrl']))}"
        )

    print("\nClemens volume-derived summary:")
    for row in clemens_volume_summaries:
        print(
            f"  {row['condition']}:"
            f" lineage_volume_mean={format_number(float(row['mean_lineage_volume_um3']))}"
            f" | lineage_proj_area_mean={format_number(float(row['mean_lineage_projected_area_um2']))}"
            f" | avg_nb_volume_mean={format_number(float(row['mean_avg_neuroblast_volume_um3']))}"
            f" | avg_nb_proj_area_mean={format_number(float(row['mean_avg_neuroblast_projected_area_um2']))}"
            f" | N={int(row['n_lineages_in_summary'])}"
        )

    print("\nSaved:")
    print(f"  {per_lineage_csv}")
    print(f"  {clemens_summary_csv}")
    print(f"  {comparison_csv}")
    print(f"  {tests_csv}")
    print(f"  {volume_per_lineage_csv}")
    print(f"  {volume_summary_csv}")
    print(f"  {volume_comparison_csv}")


if __name__ == "__main__":
    main()
