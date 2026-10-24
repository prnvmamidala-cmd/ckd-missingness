#!/usr/bin/env python3
"""Realized deletion rates of each scenario on the full cohort (one mask each, seed SEED), by income group: share of the 13
deletable cells deleted and share of participants with >= 1 deleted value. Output results/realized_rates.csv."""
import numpy as np, pandas as pd
from config import RESULTS, SEED, FEATS
from common import load_cohort, load_all, empirical_pools, SCENARIOS, make_mask, DELETABLE
df = load_cohort(); pools = empirical_pools(load_all()); g = df.pir_tier.to_numpy(); idx = [FEATS.index(f) for f in DELETABLE]
rows = []
for s, spec in SCENARIOS.items():
    D = make_mask(spec, g, SEED, pools)
    for t in ["low", "mid", "high"]:
        m = g == t; rows.append([s, t, round(100 * D[m][:, idx].mean(), 1), round(100 * D[m][:, idx].any(1).mean(), 1),
                                 round(100 * D[m][:, [FEATS.index("age"), FEATS.index("female")]].mean(), 1)])
out = pd.DataFrame(rows, columns=["scenario", "income", "cells_deleted_pct", "participants_any_deleted_pct", "age_sex_deleted_pct"])
out.to_csv(RESULTS / "realized_rates.csv", index=False); print(out.to_string())
