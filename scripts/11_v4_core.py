#!/usr/bin/env python3
"""v4 core analysis (review points A, B, D).
RUN=point : full sample, N_SPLITS (20) stratum-preserving splits x N_MASKS (10) masks -> point estimates (the estimand is the
            split-and-mask-averaged change in survey-weighted error rates for this pipeline in this population).
RUN=boot  : hierarchical outer bootstrap. Replicate b resamples PSUs with replacement within every stratum of the FULL
            development sample, then redoes a 4-fold stratum cross-fit (every stratum tested once; models, imputers and
            thresholds refit in every fold) with N_MASKS_BOOT (2) masks per fold and scenario (inner randomness averaged).
            Written incrementally (results/v4_boot/rep_XXXX.pkl) so an interrupted run resumes.
RUN=inner : N_INNER (10) cross-fit evaluations of the ORIGINAL sample with different fold/mask seeds; their variance
            estimates the inner (fold-assignment + mask) Monte Carlo variance contained in each bootstrap replicate.
Env: RUN, N_SPLITS, N_MASKS, B, N_MASKS_BOOT, N_JOBS, B_START, N_INNER."""
import os, time, pickle, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from config import RESULTS, SEED, ROOT
from common import load_cohort, load_all, empirical_pools
from v4lib import run_split, run_crossfit, psu_resample
warnings.filterwarnings("ignore")
RUN = os.environ.get("RUN", "point"); NJ = int(os.environ.get("N_JOBS", 3))
# H2 add-on (v4): SCEN=S11 runs only the preregistered uniform control with the non-stochastic repairs; outputs get suffix _h2
H2 = os.environ.get("SCEN", "") == "S11"; KW = dict(scen=("S11_cell_unif_all15",), stochastic=False) if H2 else {}; SUF = "_h2" if H2 else ""
# CE add-on (v4): SCEN=S0CE reruns S0 with non-stochastic repairs only and IPW_DET=1 so that the preregistered CE repair also gets
# selection-weighted and alternative-threshold results (same split, mask and bootstrap seeds as the main run)
if os.environ.get("SCEN", "") == "S0CE":
    os.environ["IPW_DET"] = "1"; KW = dict(scen=("S7_cell_all15",), stochastic=False); SUF = "_s0ce"
def cohort():
    df = load_cohort(); ipw = pd.read_csv(ROOT / "data" / "ipw_weights.csv")
    df = df.merge(ipw[["SEQN", "w_ipw"]], on="SEQN", how="left"); assert df.w_ipw.notna().all(); return df
def point_job(r, df, pools, K):
    t = time.time(); o = run_split(df, SEED + 1000 * r, K, pools=pools, keep_masks=True, **KW); o["split"] = r
    print(f"split {r} {time.time() - t:.0f}s", flush=True); return o
def boot_job(b, df, pools, K, outdir):
    f = outdir / f"rep_{b:04d}.pkl"
    if f.exists(): return
    t = time.time(); rng = np.random.default_rng(SEED + 7_000_000 + b)
    o = run_crossfit(psu_resample(df, rng), SEED + 7_000_000 + 1000 * b, K, pools=pools, **KW); o["rep"] = b
    pickle.dump(o, open(f, "wb")); print(f"rep {b} {time.time() - t:.0f}s", flush=True)
if __name__ == "__main__":
    df = cohort(); pools = empirical_pools(load_all())
    if RUN == "point":
        R = int(os.environ.get("N_SPLITS", 20)); K = int(os.environ.get("N_MASKS", 10))
        res = Parallel(n_jobs=NJ)(delayed(point_job)(r, df, pools, K) for r in range(R))
        pickle.dump(res, open(RESULTS / f"v4_point_raw{SUF}.pkl", "wb")); print("saved point", SUF)
    elif RUN == "boot":
        B = int(os.environ.get("B", 200)); K = int(os.environ.get("N_MASKS_BOOT", 4)); b0 = int(os.environ.get("B_START", 0))
        outdir = RESULTS / f"v4_boot{SUF}"; outdir.mkdir(exist_ok=True)
        Parallel(n_jobs=NJ)(delayed(boot_job)(b, df, pools, K, outdir) for b in range(b0, B))
        print("boot done")
    if RUN == "inner":
        K = int(os.environ.get("N_MASKS_BOOT", 2)); NI = int(os.environ.get("N_INNER", 10)); outdir = RESULTS / f"v4_inner{SUF}"; outdir.mkdir(exist_ok=True)
        def inner_job(i):
            f = outdir / f"inner_{i:03d}.pkl"
            if f.exists(): return
            t = time.time(); o = run_crossfit(df, SEED + 9_000_000 + 1000 * i, K, pools=pools, **KW); pickle.dump(o, open(f, "wb")); print(f"inner {i} {time.time() - t:.0f}s", flush=True)
        Parallel(n_jobs=NJ)(delayed(inner_job)(i) for i in range(NI)); print("inner done")
