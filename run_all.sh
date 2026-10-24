#!/usr/bin/env bash
# Full pipeline (v3 analyses reused for exploratory results + v4 core analyses and inference).
set -euo pipefail
cd "$(dirname "$0")/scripts"
python 00_download_nhanes.py
python 01_harmonize_nhanes.py
python 02_descriptives.py
RUN=main N_SPLITS=20 N_MASKS=20 N_BOOT=100 N_JOBS=${N_JOBS:-5} python 03_repeated_splits.py
RUN=ablation N_SPLITS=20 N_MASKS=10 N_BOOT=100 N_JOBS=${N_JOBS:-5} python 03_repeated_splits.py
python 04_external.py
python 05_aggregate.py
N_JOBS=${N_JOBS:-5} python 07_decision_curve.py
python 08_tables.py
python 09_realized_rates.py
# ---- v4 ----
python 10_selection_weights.py
RUN=point N_SPLITS=20 N_MASKS=10 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
RUN=boot B=120 N_MASKS_BOOT=2 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
RUN=inner N_INNER=12 N_MASKS_BOOT=2 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
SCEN=S11 RUN=point N_SPLITS=20 N_MASKS=10 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
SCEN=S11 RUN=boot B=120 N_MASKS_BOOT=2 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
SCEN=S11 RUN=inner N_INNER=12 N_MASKS_BOOT=2 N_JOBS=${N_JOBS:-3} python 11_v4_core.py
python 12_coverage_sim.py
python 13_v4_aggregate.py
python 13_v4_aggregate.py _h2
python 14_v4_tables.py
python 15_v4_figures.py
