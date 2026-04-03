# Run from the project root: make [target]
PYTHON := .venv/bin/python

# ── Output sentinels ───────────────────────────────────────────────────────────
SIM_NPZ         := data/sim/sim_geo_counts_endpoints.npz
EXP_WT_NPZ      := data/exp/npz_files/exp_wt_npz/lobe1_wt_maxproj.npz
WRL_METRICS_CSV := data/exp/wrl_metrics/experimental_wrl_metrics_per_lineage.csv
WRL_PLOTS       := data/exp/plots/hist_dpn_counts_wt_vs_mud.png
CLEMENS_CSV     := data/exp/from_clemens/clemens_cell_count_per_lineage.csv
CLEMENS_PLOTS   := data/exp/from_clemens/plots/wt_vs_mud_stats_wrl_vs_clemens.csv
DETDIFF_PLOTS   := data/sim/bioparams_analysis/compare_detdiff_nb_count.png
VIZ_INDEX       := data/exp/viz_cache/lineage_index.csv
ROBUSTNESS_CSV  := data/exp/robustness/robustness_summary.csv

# ── Top-level targets ──────────────────────────────────────────────────────────
.PHONY: all process wrl-metrics wrl-plots clemens detdiff sim-vs-wrl viz robustness clean

## Run all non-interactive targets in dependency order.
all: process wrl-metrics wrl-plots clemens detdiff robustness

## Preprocess raw data into NPZ files (sim + exp WRL → npz).
process: $(SIM_NPZ) $(EXP_WT_NPZ)

## Compute per-lineage WRL metrics CSVs from experimental WRL files.
wrl-metrics: $(WRL_METRICS_CSV)

## Plot WT vs mudmut histograms directly from WRL meshes.
wrl-plots: $(WRL_PLOTS)

## Extract Clemens spreadsheet metrics and plot vs WRL (requires wrl-metrics).
clemens: $(CLEMENS_CSV) $(CLEMENS_PLOTS)

## Plot sim conditions vs sim bioparameters (det-diff comparison).
detdiff: $(DETDIFF_PLOTS)

## Plot sim conditions vs experimental WRL data (interactive — not in `all`).
sim-vs-wrl: $(SIM_NPZ) $(EXP_WT_NPZ)
	$(PYTHON) -m scripts.plot_sim_conditions_vs_wrl

## Build lineage mesh cache for the 3D visualizer (not in `all`).
viz: $(VIZ_INDEX)

## Run robustness analysis on WT vs mudmut differences.
robustness: $(ROBUSTNESS_CSV)

## View a lineage: make view-lineage ROW=7  or  make view-lineage ID=lin0007
view-lineage:
	$(PYTHON) -m scripts.visualize_lineage $(if $(ROW),--row $(ROW),) $(if $(ID),--id $(ID),--list)

# ── Rules ─────────────────────────────────────────────────────��────────────────

# Sim preprocessing: raw JSON → sim_geo_counts_endpoints.npz
$(SIM_NPZ): $(wildcard data/sim/sim*/*.CELLS.json)
	$(PYTHON) -m scripts.process_simdata

# Exp preprocessing: raw WRL → per-lobe NPZ files
$(EXP_WT_NPZ): $(wildcard data/exp/wrl_files/**/*.wrl)
	$(PYTHON) -m scripts.process_lobe_maxproj

# WRL metrics: raw WRL → per-lineage and summary CSVs
$(WRL_METRICS_CSV): $(wildcard data/exp/wrl_files/**/*.wrl)
	$(PYTHON) -m scripts.extract_wrl_lineage_metrics

# WRL plots: raw WRL → histogram PNGs
$(WRL_PLOTS): $(wildcard data/exp/wrl_files/**/*.wrl)
	$(PYTHON) -m scripts.plot_exp_wrl_histograms

# Clemens CSVs: Clemens Excel + WRL metrics → comparison CSVs
$(CLEMENS_CSV): $(WRL_METRICS_CSV) $(wildcard data/exp/from_clemens/*.xlsx)
	$(PYTHON) -m scripts.extract_clemens_lineage_metrics

# Clemens plots: Clemens Excel + WRL per-lineage CSV → comparison plots
$(CLEMENS_PLOTS): $(WRL_METRICS_CSV) $(wildcard data/exp/from_clemens/*.xlsx)
	$(PYTHON) -m scripts.plot_exp_wrl_vs_clemens

# Detdiff plots: raw sim JSONs → bioparams comparison plots
$(DETDIFF_PLOTS): $(wildcard data/sim/sim*/*.CELLS.json)
	$(PYTHON) -m scripts.compare_detdiff

# Viz cache: raw WRL → lineage index + per-lineage NPZ meshes
$(VIZ_INDEX): $(wildcard data/exp/wrl_files/**/*.wrl)
	$(PYTHON) -m scripts.preprocess_lineages_for_viz

# Robustness analysis: per-lineage CSVs → bootstrap/permutation/subsample results
$(ROBUSTNESS_CSV): $(WRL_METRICS_CSV) $(CLEMENS_CSV)
	$(PYTHON) -m scripts.robustness_analysis

## Delete all generated output files so targets will be rebuilt from scratch.
clean:
	rm -rf data/sim/sim_geo_counts_endpoints.npz
	rm -rf data/exp/npz_files/
	rm -rf data/exp/wrl_metrics/
	rm -rf data/exp/plots/
	rm -rf data/exp/from_clemens/plots/
	rm -f data/exp/from_clemens/clemens_*.csv
	rm -rf data/sim/bioparams_analysis/
	rm -rf data/exp/viz_cache/
	rm -rf data/exp/robustness/
