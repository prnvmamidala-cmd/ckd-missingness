#!/usr/bin/env python3
"""Aggregate per-split results: point estimate = mean over splits; 95% CI = 2.5/97.5 percentiles of the mixture of all
split-specific bootstrap replicates (each replicate already resamples PSUs within test strata and masks), so the interval
includes sampling, mask and split-to-split variability. Also reports the split-to-split SD/range of the point estimates.
Outputs results/v3_summary.json, results/v3_ablation_summary.json (if present) and CSV tables."""
import json, pickle
import numpy as np, pandas as pd
from config import RESULTS, FEATS
def load(fn): return pickle.load(open(RESULTS / fn, "rb"))
def pooled(items):
    est = np.array([e for e, _ in items], float); reps = np.concatenate([np.asarray(r, float) for _, r in items])
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    # within-split-only interval (split variability removed) for comparison
    cen = np.concatenate([np.asarray(r, float) - e for e, r in items]) + est.mean()
    wlo, whi = np.nanpercentile(cen, [2.5, 97.5])
    return dict(est=float(est.mean()), lo=float(lo), hi=float(hi), split_sd=float(est.std(ddof=1)) if len(est) > 1 else 0.0,
                split_min=float(est.min()), split_max=float(est.max()), within_lo=float(wlo), within_hi=float(whi), n_splits=len(est))
def summarize(R):
    S = R["splits"]; out = {"meta": R["meta"]}
    out["design"] = {k: [s[k] for s in S] for k in ["n", "n_strata", "n_psu_test", "test_ckd"]}
    out["design_mean"] = {"test_n": float(np.mean([s["n"]["test"] for s in S])), "test_psu": float(np.mean([s["n_psu_test"] for s in S])),
                          "test_strata": float(np.mean([s["n_strata"]["test"] for s in S])),
                          "test_ckd": {t: float(np.mean([s["test_ckd"][t] for s in S])) for t in ["low", "mid", "high", "all"]},
                          "train_n": float(np.mean([s["n"]["train"] for s in S])), "cal_n": float(np.mean([s["n"]["cal"] for s in S]))}
    out["thresholds"] = {m: {th: [s["thresholds"][m][th] for s in S] for th in S[0]["thresholds"][m]} for m in S[0]["thresholds"]}
    out["complete"] = {"|".join(k): {m: pooled([s["complete"][k][m] for s in S]) for m in S[0]["complete"][k]} for k in S[0]["complete"]}
    out["keys"] = {}
    for k in S[0]["keys"]:
        d = {}
        for m, v in S[0]["keys"][k].items():
            if isinstance(v, tuple): d[m] = pooled([s["keys"][k][m] for s in S])
            elif m == "dFNR_low_per_mask":
                d["mask_sd_within_split"] = float(np.mean([np.std(s["keys"][k][m], ddof=1) for s in S]))
                d["frac_masks_positive"] = float(np.mean([np.mean(np.asarray(s["keys"][k][m]) > 0) for s in S]))
                d["split_estimates_dFNR_low"] = [float(s["keys"][k]["dFNR_low"][0]) for s in S]
            else: d[m] = float(np.mean([s["keys"][k][m] for s in S]))
        out["keys"]["|".join(k)] = d
    out["auroc"] = {"|".join(k): {t: float(np.nanmean([s["auroc"][k][t] for s in S])) for t in S[0]["auroc"][k]} for k in S[0]["auroc"]}
    if False and S[0].get("dca"):   # raw-score curves are not calibrated; decision curves come from 07_decision_curve.py
        out["dca"] = {"|".join(k): {t: {"mean": np.mean([s["dca"][k][t] for s in S], 0).tolist(),
                                        "p2.5": np.percentile([s["dca"][k][t] for s in S], 2.5, 0).tolist(),
                                        "p97.5": np.percentile([s["dca"][k][t] for s in S], 97.5, 0).tolist()} for t in S[0]["dca"][k]} for k in S[0]["dca"]}
    if S[0].get("mech"):
        fe = R["meta"]["feats"]; mech = {}
        for sc in S[0]["mech"]:
            for meth in S[0]["mech"][sc]:
                agg = {}
                for f in ["n_flip", "n_stay", "dshap_flip", "dshap_stay", "shift_flip", "cnt_flip", "shift_stay", "cnt_stay"]:
                    agg[f] = np.sum([np.asarray(s["mech"][sc][meth][f], float) for s in S], 0)
                mech[f"{sc}|{meth}"] = {"n_flip_events": int(agg["n_flip"]), "n_stay_events": int(agg["n_stay"]),
                    "mean_dSHAP_flipped": dict(zip(fe, (agg["dshap_flip"] / max(agg["n_flip"], 1)).tolist())),
                    "mean_dSHAP_stayed": dict(zip(fe, (agg["dshap_stay"] / max(agg["n_stay"], 1)).tolist())),
                    "mean_risk_signed_shift_deleted_flipped": dict(zip(fe, (agg["shift_flip"] / np.maximum(agg["cnt_flip"], 1)).tolist())),
                    "mean_risk_signed_shift_deleted_stayed": dict(zip(fe, (agg["shift_stay"] / np.maximum(agg["cnt_stay"], 1)).tolist())),
                    "n_deleted_cells_flipped": dict(zip(fe, agg["cnt_flip"].astype(int).tolist()))}
        out["mech"] = mech
    if S[0].get("single"):
        out["single"] = {"|".join(k): dict(est=float(np.mean([s["single"][k] for s in S])), p2_5=float(np.percentile([s["single"][k] for s in S], 2.5)),
                                           p97_5=float(np.percentile([s["single"][k] for s in S], 97.5))) for k in S[0]["single"]}
    if S[0].get("inter"):
        keys = set().union(*[s["inter"].keys() for s in S])
        out["inter"] = {"|".join(k): pooled([s["inter"][k] for s in S if k in s["inter"]]) for k in keys}
        out["inter_n_mean"] = {k: float(np.mean([s["inter_n"][k] for s in S])) for k in S[0]["inter_n"]}
    if S[0].get("aware_thresholds"):
        out["aware_thresholds_mean"] = {sc: {a: {th: float(np.mean([s["aware_thresholds"][sc][a][th] for s in S])) for th in S[0]["aware_thresholds"][sc][a]}
                                         for a in S[0]["aware_thresholds"][sc]} for sc in S[0]["aware_thresholds"]}
    return out
if __name__ == "__main__":
    out = summarize(load("v3_splits_raw.pkl"))
    json.dump(out, open(RESULTS / "v3_summary.json", "w"), indent=1)
    if (RESULTS / "v3_ablation_raw.pkl").exists():
        json.dump(summarize(load("v3_ablation_raw.pkl")), open(RESULTS / "v3_ablation_summary.json", "w"), indent=1)
    f = lambda v: f"{100*v['est']:.1f} ({100*v['lo']:.1f}, {100*v['hi']:.1f})"
    for sc in ["S1_cell", "S2_block", "S3_empirical", "S4_emp_pooled", "S5_cell_unif", "S6_block_unif", "S7_cell_all15", "S8_lab_only", "S9_exam_only", "S10_quest_only"]:
        for meth in ["mean", "median", "knn", "mice", "xgb_native", "xgb_aware_native", "xgb_aware_ind"]:
            k = f"{sc}|xgb|{meth}|sens80"
            if k in out["keys"]:
                d = out["keys"][k]; print(k, "low", f(d["dFNR_low"]), "high", f(d["dFNR_high"]), "gap", f(d["gap"]), "dFPRlow", f(d["dFPR_low"]), "splitSD %.1f" % (100*d["dFNR_low"]["split_sd"]))
