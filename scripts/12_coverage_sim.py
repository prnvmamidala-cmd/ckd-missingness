#!/usr/bin/env python3
"""v4 small coverage simulation for the outer PSU bootstrap (review point A). It checks ONLY the design-based part of the
interval: a fixed (pre-trained) XGBoost model, fixed threshold and mean imputation, under block deletion S2. The model and
imputer are NOT refit inside replicates, so coverage of the model-refitting component is not checked.
Superpopulation: each of the 45 strata gets M = 10 pseudo-PSUs; a pseudo-PSU is a with-replacement resample of one of the
stratum's two real PSUs (chosen at random). A "sample" takes 2 pseudo-PSUs per stratum (as NHANES does). The statistic
mimics the real estimand: survey-weighted low-income dFNR (and low-minus-high gap) on a random 12-stratum test subset,
averaged over 20 subsets and 10 masks (point estimate). Truth = the same quantity over the whole superpopulation.
Each bootstrap replicate (B draws of 2 PSUs with replacement per stratum) is evaluated like the real analysis: one 4-fold
stratum cross-fit x 2 masks. Intervals: (i) naive percentile; (ii) sqrt(2)-rescaled ("conservative", the v4 primary method); (iii) sqrt(2)-rescaled normal interval
after subtracting the inner (subset + mask) Monte Carlo variance and adding the Monte Carlo variance of the point estimate. Output results/v4_coverage_sim.json."""
import os, json, warnings
import numpy as np
from config import RESULTS, SEED, FEATS
from common import load_cohort, SCENARIOS, make_mask, apply_mask, TypedSimpleImputer, thr_sens
from v4lib import xgb
warnings.filterwarnings("ignore")
R = int(os.environ.get("R", 300)); B = int(os.environ.get("B", 200)); Q = 40; M = 10; NT = 12; KM = 2; KPT = 10; NSPLIT_PT = 20
rng = np.random.default_rng(SEED + 42)
df = load_cohort(); X = df[FEATS].to_numpy(float); y = df.y.to_numpy(int); w = df.w.to_numpy(float); g = df.pir_tier.to_numpy()
pos = np.average(y, weights=w); spw = (1 - pos) / pos
mdl = xgb(spw, SEED).fit(X, y, sample_weight=w / w.mean()); p0 = mdl.predict_proba(X)[:, 1]; thr = thr_sens(y, p0, w)
imp = TypedSimpleImputer("mean").fit(X)
miss0 = (p0 < thr).astype(float)
bank = np.zeros((Q, len(df)), np.float32)
for q in range(Q):
    D = make_mask(SCENARIOS["S2_block"], g, SEED + 9000 + q); bank[q] = mdl.predict_proba(imp.transform(apply_mask(X, D)))[:, 1] < thr
strata = df.strata.to_numpy(); psu = df.psu.to_numpy(); S = np.unique(strata)
pop = []   # list of (stratum index, member index array)
for si, s in enumerate(S):
    ps = np.unique(psu[strata == s])
    for _ in range(M):
        mem = np.flatnonzero(psu == rng.choice(ps)); pop.append((si, rng.choice(mem, size=len(mem), replace=True)))
pop_s = np.array([a for a, _ in pop])
lowpos = ((g == "low") & (y == 1)).astype(float); highpos = ((g == "high") & (y == 1)).astype(float)
def _fold_stat(units, us, test, rng, n_mask, expect):
    idx = np.concatenate([pop[u][1] for u, s in zip(units, us) if s in test])
    ww = w[idx]; lp = lowpos[idx] * ww; hp = highpos[idx] * ww
    if expect: mi = bank[:, idx].mean(0)
    else: mi = bank[rng.integers(0, Q, size=(n_mask, len(idx))), idx[None, :]].mean(0)
    dl = (lp * (mi - miss0[idx])).sum() / lp.sum(); dh = (hp * (mi - miss0[idx])).sum() / hp.sum()
    return dl, dl - dh
def stat(units, rng, n_sub, n_mask, expect=False):
    """Point-estimate analogue: average over n_sub random 12-stratum test subsets (random splits) x n_mask masks."""
    us = pop_s[units]
    return np.mean([_fold_stat(units, us, set(rng.choice(len(S), NT, replace=False)), rng, n_mask, expect) for _ in range(n_sub)], 0)
def stat_cf(units, rng, n_mask):
    """Bootstrap-replicate analogue: one random 4-fold partition of the 45 strata (every stratum tested once) x n_mask masks."""
    us = pop_s[units]; perm = rng.permutation(len(S))
    return np.mean([_fold_stat(units, us, set(perm[k::4]), rng, n_mask, False) for k in range(4)], 0)
allunits = np.arange(len(pop)); truth = stat(allunits, rng, 3000, 0, expect=True)
res = {k: [] for k in ["naive", "cons", "corr"]}; diag = {k: [] for k in ["theta", "se_corr", "sd_boot_x_sqrt2", "inner_sd"]}; widths = {k: [] for k in res}
for r in range(R):
    samp = np.concatenate([rng.choice(np.flatnonzero(pop_s == si), 2, replace=False) for si in range(len(S))])
    by_s = [samp[pop_s[samp] == si] for si in range(len(S))]
    th = stat(samp, rng, NSPLIT_PT, KPT)                                  # point estimate: 20 random splits x 10 masks
    v_pt = np.var([stat(samp, rng, 1, KPT) for _ in range(20)], axis=0, ddof=1) / NSPLIT_PT   # MC variance of the point estimate
    inner = np.var([stat_cf(samp, rng, KM) for _ in range(12)], axis=0, ddof=1)               # inner variance of one replicate
    bs = np.array([stat_cf(np.concatenate([rng.choice(u, 2, replace=True) for u in by_s]), rng, KM) for _ in range(B)])
    m, q = bs.mean(0), np.percentile(bs, [2.5, 97.5], axis=0)
    ci = {"naive": (th + (q[0] - m), th + (q[1] - m)), "cons": (th + np.sqrt(2) * (q[0] - m), th + np.sqrt(2) * (q[1] - m))}
    se = np.sqrt(2 * np.maximum(bs.var(0, ddof=1) - inner, 0) + v_pt); ci["corr"] = (th - 1.96 * se, th + 1.96 * se)
    diag["theta"].append(th); diag["se_corr"].append(se); diag["sd_boot_x_sqrt2"].append(np.sqrt(2 * bs.var(0, ddof=1))); diag["inner_sd"].append(np.sqrt(inner))
    for k, (lo, hi) in ci.items(): res[k].append((lo <= truth) & (truth <= hi)); widths[k].append(hi - lo)
    if r % 25 == 0: print(r, flush=True)
out = {"R": R, "B": B, "truth_dFNR_low_pp": 100 * float(truth[0]), "truth_gap_pp": 100 * float(truth[1]),
       "coverage": {k: {"dFNR_low": float(np.mean([v[0] for v in res[k]])), "gap": float(np.mean([v[1] for v in res[k]]))} for k in res},
       "mean_width_pp": {k: {"dFNR_low": 100 * float(np.mean([v[0] for v in widths[k]])), "gap": 100 * float(np.mean([v[1] for v in widths[k]]))} for k in res},
       "mc_se_of_coverage_at_0.95": float(np.sqrt(0.95 * 0.05 / R)),
       "diagnostics_pp": {"mean_estimate_minus_truth": (100 * (np.mean(diag["theta"], 0) - truth)).tolist(),
                          "empirical_sd_of_estimate": (100 * np.std(diag["theta"], 0, ddof=1)).tolist(),
                          "mean_corrected_se": (100 * np.mean(diag["se_corr"], 0)).tolist(),
                          "mean_sqrt2_boot_sd": (100 * np.mean(diag["sd_boot_x_sqrt2"], 0)).tolist(),
                          "mean_inner_sd_1split_4masks": (100 * np.mean(diag["inner_sd"], 0)).tolist()}}
json.dump(out, open(RESULTS / "v4_coverage_sim.json", "w"), indent=1); print(json.dumps(out, indent=1))
