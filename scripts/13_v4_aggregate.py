#!/usr/bin/env python3
"""Aggregate v4 point estimates (20 random stratum-preserving splits x 10 masks on the full sample) and the hierarchical
outer PSU bootstrap (B replicates; each = PSU resample within strata of the full sample + 4-fold stratum cross-fit x 2 masks).
Intervals per estimate theta_hat (point estimate) and bootstrap replicates theta*_b:
  primary  "conservative": theta_hat + sqrt(2) * (2.5th, 97.5th percentile of theta* - mean theta*)
           sqrt(2) = sqrt(n_h/(n_h-1)) with n_h = 2 PSUs per stratum corrects the known downward bias of the with-replacement
           PSU bootstrap variance (Rao & Wu 1988). Inner (fold + mask) Monte Carlo noise inside each replicate is NOT
           removed, so the interval is expected to be conservative (it was in the coverage simulation).
  secondary "corrected": theta_hat +/- 1.96 * sqrt(2 * max(Var(theta*) - V_inner, 0) + V_point), V_inner = variance of 12
           cross-fit evaluations of the original sample, V_point = Monte Carlo variance of theta_hat (split variance / 20).
           It under-covered in the coverage simulation and is reported for comparison only.
Output: results/v4_summary.json (or v4_summary_h2.json with argument _h2)"""
import json, pickle, glob
import numpy as np
from config import RESULTS
from v4lib import STATS
import sys
SUF = sys.argv[1] if len(sys.argv) > 1 else ""      # "" (main) or "_h2" (preregistered uniform control)
P = pickle.load(open(RESULTS / f"v4_point_raw{SUF}.pkl", "rb"))
BOOT = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(RESULTS / f"v4_boot{SUF}" / "rep_*.pkl")))]
INNER = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(str(RESULTS / f"v4_inner{SUF}" / "inner_*.pkl")))]
V3 = json.load(open(RESULTS / "v3_summary.json"))["keys"]
V3M = {"sii": "mice", "native": "xgb_native"}
R2 = np.sqrt(2)
def interval(th, bs, v_in, v_pt):
    bs = np.asarray(bs, float); m = bs.mean(); q = np.percentile(bs, [2.5, 97.5])
    se = np.sqrt(2 * max(bs.var(ddof=1) - v_in, 0) + v_pt) if v_in is not None else np.nan
    return {"est": th, "lo": th + R2 * (q[0] - m), "hi": th + R2 * (q[1] - m), "naive_lo": th + (q[0] - m), "naive_hi": th + (q[1] - m),
            "corr_lo": th - 1.96 * se, "corr_hi": th + 1.96 * se, "boot_sd_x_sqrt2": R2 * bs.std(ddof=1), "n_boot": len(bs),
            "frac_boot_gt0": float((bs > 0).mean())}
out = {"meta": {"n_splits_point": len(P), "n_masks_point": int(P[0]["est"][next(iter(P[0]["est"]))].shape[0]), "B": len(BOOT),
                "n_inner": len(INNER), "stats": STATS}, "keys": {}, "complete": {}, "design_point": {}}
out["design_point"] = {"test_n_mean": float(np.mean([p["n"]["test"] for p in P])), "train_n_mean": float(np.mean([p["n"]["train"] for p in P])),
                       "cal_n_mean": float(np.mean([p["n"]["cal"] for p in P])),
                       "ckd_test_mean": {t: float(np.mean([p["ckd_test"][t] for p in P])) for t in P[0]["ckd_test"]}}
for key in P[0]["est"]:
    arr = np.array([p["est"][key] for p in P])           # splits x masks x stats
    sm = arr.mean(1); th = sm.mean(0); msd = np.sqrt(arr.var(1, ddof=1).mean(0))
    have_b = key in BOOT[0]["est"] if BOOT else False
    d = {}
    for j, s in enumerate(STATS):
        if have_b:
            bs = [b["est"][key][j] for b in BOOT]; vin = np.var([i["est"][key][j] for i in INNER], ddof=1) if len(INNER) > 1 else None
            d[s] = interval(float(th[j]), bs, vin, float(sm[:, j].var(ddof=1) / len(P)))
        else: d[s] = {"est": float(th[j])}
        d[s].update({"split_sd": float(sm[:, j].std(ddof=1)), "split_min": float(sm[:, j].min()), "split_max": float(sm[:, j].max()), "mask_sd": float(msd[j])})
        k3 = f"{key[0]}|{key[1]}|{V3M.get(key[2], key[2])}|{key[3]}"
        if key[4] == "w" and k3 in V3 and s in V3[k3]: d[s]["v3"] = {x: V3[k3][s][x] for x in ("est", "lo", "hi")}
    out["keys"]["|".join(key)] = d
for key in P[0]["complete"]:
    d = {}
    for s in P[0]["complete"][key]:
        v = np.array([p["complete"][key][s] for p in P]); th = float(v.mean())
        if BOOT and key in BOOT[0]["complete"]:
            vin = np.var([i["complete"][key][s] for i in INNER], ddof=1) if len(INNER) > 1 else None
            d[s] = interval(th, [b["complete"][key][s] for b in BOOT], vin, float(v.var(ddof=1) / len(P)))
        else: d[s] = {"est": th}
        d[s].update({"split_min": float(v.min()), "split_max": float(v.max())})
    out["complete"]["|".join(key)] = d
json.dump(out, open(RESULTS / f"v4_summary{SUF}.json", "w"), indent=1, default=float)
f = lambda v: f"{100 * v['est']:.1f} ({100 * v['lo']:.1f}, {100 * v['hi']:.1f})" if "lo" in v else f"{100 * v['est']:.1f}"
for k, d in out["keys"].items():
    if k.split("|")[1] == "xgb" and k.split("|")[3] == "sens80":
        print(k.ljust(38), "low", f(d["dFNR_low"]), "| high", f(d["dFNR_high"]), "| gap", f(d["gap"]), "| dFPRlow", f(d["dFPR_low"]),
              "| v3", (lambda v: f"{100*v['est']:.1f} ({100*v['lo']:.1f}, {100*v['hi']:.1f})")(d["dFNR_low"]["v3"]) if "v3" in d["dFNR_low"] else "")
