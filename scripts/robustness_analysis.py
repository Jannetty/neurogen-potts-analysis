"""
robustness_analysis.py

Tests how robust the WT vs mudmut differences are for three metrics:
  - lineage_volume_um3
  - avg_neuroblast_volume_um3
  - total_cell_count

Runs independently on two datasets:
  1. WRL pipeline data  (data/exp/wrl_metrics/experimental_wrl_metrics_per_lineage.csv)
  2. Clemens data       (data/exp/from_clemens/clemens_volume_per_lineage.csv)

For each dataset × metric:
  - Observed Mann-Whitney U test + rank-biserial effect size
  - Bootstrap CI on effect size (n_boot iterations, with replacement)
  - Permutation test p-value (n_perm label shuffles)
  - Subsample size curve: at each fraction, run the test n_sub times and
    record the fraction of runs that are significant (p < alpha)

Outputs
-------
data/exp/robustness/robustness_summary.csv
    One row per dataset × metric with observed stats and bootstrap CIs.

data/exp/robustness/plots/
    subsample_curve_{dataset}_{metric}.png   – fraction-significant vs subsample %
    bootstrap_effect_{dataset}_{metric}.png  – bootstrap distribution of effect size
    raw_data_{dataset}_{metric}.png          – strip/violin plots of raw values
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

METRICS = {
    "lineage_volume_um3":       "Lineage volume (µm³)",
    "avg_neuroblast_volume_um3":"Avg neuroblast volume (µm³)",
    "total_cell_count":         "Total cell count",
}

DATASETS = {
    "wrl": {
        "path": Path("data/exp/wrl_metrics/experimental_wrl_metrics_per_lineage.csv"),
        "label": "WRL pipeline",
        "condition_col": "condition",
        "wt_label": "WT",
        "mut_label": "mudmut",
    },
    "clemens": {
        "path": Path("data/exp/from_clemens/clemens_volume_per_lineage.csv"),
        "label": "Clemens",
        "condition_col": "condition",
        "wt_label": "WT",
        "mut_label": "mudmut",
    },
}

WT_COLOR  = "#4C72B0"
MUT_COLOR = "#DD8452"
ALPHA = 0.05


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation: effect size for Mann-Whitney U.
    Range [-1, 1]; positive means group a tends to be larger than group b.
    """
    n_a, n_b = len(a), len(b)
    u_stat, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 1 - (2 * u_stat) / (n_a * n_b)


def mwu_pvalue(a: np.ndarray, b: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(p)


def bootstrap_effect_size(
    a: np.ndarray, b: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    effects = np.empty(n_boot)
    for i in range(n_boot):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        effects[i] = rank_biserial(a_s, b_s)
    return effects


def permutation_pvalue(
    a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator
) -> float:
    observed = abs(rank_biserial(a, b))
    pooled = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        a_s, b_s = pooled[:n_a], pooled[n_a:]
        if abs(rank_biserial(a_s, b_s)) >= observed:
            count += 1
    return (count + 1) / (n_perm + 1)


def subsample_curve(
    a: np.ndarray,
    b: np.ndarray,
    fractions: np.ndarray,
    n_sub: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """For each fraction, return the proportion of runs where p < ALPHA."""
    frac_sig = np.empty(len(fractions))
    for fi, frac in enumerate(fractions):
        n_a = max(2, int(round(len(a) * frac)))
        n_b = max(2, int(round(len(b) * frac)))
        sig = 0
        for _ in range(n_sub):
            a_s = rng.choice(a, size=n_a, replace=False)
            b_s = rng.choice(b, size=n_b, replace=False)
            if mwu_pvalue(a_s, b_s) < ALPHA:
                sig += 1
        frac_sig[fi] = sig / n_sub
    return frac_sig


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_raw_data(a: np.ndarray, b: np.ndarray, metric_label: str,
                  dataset_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    rng = np.random.default_rng(1)
    for vals, color, label, x in [(a, WT_COLOR, "WT", 0), (b, MUT_COLOR, "mudmut", 1)]:
        jitter = rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals,
                   color=color, alpha=0.6, s=20, zorder=3)
        ax.plot([x - 0.2, x + 0.2], [np.median(vals)] * 2,
                color=color, linewidth=2.5, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "mudmut"])
    ax.set_ylabel(metric_label)
    ax.set_title(f"{dataset_label}\n{metric_label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_bootstrap(effects: np.ndarray, observed: float, ci: tuple[float, float],
                   metric_label: str, dataset_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(effects, bins=40, color=WT_COLOR, alpha=0.7, edgecolor="white")
    ax.axvline(observed, color="black", linewidth=1.5, label=f"observed = {observed:.3f}")
    ax.axvline(ci[0], color="gray", linewidth=1, linestyle="--", label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.axvline(ci[1], color="gray", linewidth=1, linestyle="--")
    ax.axvline(0, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Rank-biserial effect size (WT − mudmut)")
    ax.set_ylabel("Bootstrap count")
    ax.set_title(f"{dataset_label} — {metric_label}\nBootstrap effect size distribution", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_subsample_curve(fractions: np.ndarray, frac_sig: np.ndarray,
                          n_wt: int, n_mut: int,
                          metric_label: str, dataset_label: str, out_path: Path) -> None:
    pct = fractions * 100
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(pct, frac_sig * 100, color=WT_COLOR, linewidth=2, marker="o", markersize=4)
    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95% significance rate")
    ax.axhline(50, color="lightgray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Subsample size (% of full dataset per group)")
    ax.set_ylabel("% of runs p < 0.05")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, 105)

    # Secondary x-axis showing approximate n per group
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_pcts = np.array([25, 50, 75, 100])
    ax2.set_xticks(tick_pcts)
    ax2.set_xticklabels([
        f"n≈{int(round(p/100*n_wt))}/{int(round(p/100*n_mut))}"
        for p in tick_pcts
    ], fontsize=7)
    ax2.set_xlabel("Approx n (WT/mudmut)", fontsize=8)

    ax.set_title(f"{dataset_label} — {metric_label}\nSubsample robustness", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robustness analysis of WT vs mudmut differences."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/exp/robustness"))
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="Bootstrap iterations (default: 1000)")
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Permutation test iterations (default: 1000)")
    parser.add_argument("--n-sub", type=int, default=200,
                        help="Iterations per subsample fraction (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dir = args.out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    fractions = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])

    summary_rows: list[dict] = []

    for ds_key, ds in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds['label']}  ({ds['path']})")
        print(f"{'='*60}")

        if not ds["path"].exists():
            print(f"  SKIPPING — file not found: {ds['path']}")
            continue

        df = pd.read_csv(ds["path"])
        wt  = df[df[ds["condition_col"]] == ds["wt_label"]]
        mut = df[df[ds["condition_col"]] == ds["mut_label"]]
        print(f"  WT n={len(wt)}, mudmut n={len(mut)}")

        for metric, metric_label in METRICS.items():
            if metric not in df.columns:
                print(f"  Skipping {metric} — column not found")
                continue

            a = wt[metric].dropna().to_numpy(dtype=float)
            b = mut[metric].dropna().to_numpy(dtype=float)

            if len(a) < 2 or len(b) < 2:
                print(f"  Skipping {metric} — insufficient data")
                continue

            print(f"\n  {metric_label}")

            # Observed test
            obs_p = mwu_pvalue(a, b)
            obs_effect = rank_biserial(a, b)
            print(f"    Observed: p={obs_p:.4g}, effect={obs_effect:.3f}")

            # Bootstrap effect size CI
            boot_effects = bootstrap_effect_size(a, b, args.n_boot, rng)
            ci = (float(np.percentile(boot_effects, 2.5)),
                  float(np.percentile(boot_effects, 97.5)))
            print(f"    Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")

            # Permutation p-value
            perm_p = permutation_pvalue(a, b, args.n_perm, rng)
            print(f"    Permutation p: {perm_p:.4g}")

            # Subsample curve
            frac_sig = subsample_curve(a, b, fractions, args.n_sub, rng)
            full_sig_pct = frac_sig[-1] * 100
            half_sig_pct = frac_sig[fractions == 0.50][0] * 100 if 0.50 in fractions else float("nan")
            print(f"    Subsample sig rate: 100%={full_sig_pct:.0f}%  50%={half_sig_pct:.0f}%")

            # Plots
            slug = f"{ds_key}_{metric}"
            plot_raw_data(a, b, metric_label, ds["label"],
                          plot_dir / f"raw_data_{slug}.png")
            plot_bootstrap(boot_effects, obs_effect, ci, metric_label, ds["label"],
                           plot_dir / f"bootstrap_effect_{slug}.png")
            plot_subsample_curve(fractions, frac_sig, len(a), len(b),
                                 metric_label, ds["label"],
                                 plot_dir / f"subsample_curve_{slug}.png")

            summary_rows.append({
                "dataset":          ds["label"],
                "metric":           metric,
                "n_wt":             len(a),
                "n_mut":            len(b),
                "median_wt":        float(np.median(a)),
                "median_mut":       float(np.median(b)),
                "observed_p":       obs_p,
                "perm_p":           perm_p,
                "effect_size":      obs_effect,
                "boot_ci_lo":       ci[0],
                "boot_ci_hi":       ci[1],
                "pct_sig_at_100pct": full_sig_pct,
                "pct_sig_at_50pct":  half_sig_pct,
            })

    # Summary CSV
    summary_path = args.out_dir / "robustness_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written to {summary_path}")
    print(f"Plots written to   {plot_dir}/")


if __name__ == "__main__":
    main()
