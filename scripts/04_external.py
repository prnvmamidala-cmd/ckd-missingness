#!/usr/bin/env python3
"""Temporal external validation (exploratory): models, imputers and thresholds developed on NHANES 2013-2018 only
(stratum-grouped 75/25 train/calibration split) are applied unchanged to complete-case adults from NHANES Aug 2021-Aug 2023
(a separate sample; no participant overlap). Masks (K) are applied to the 2021-2023 data; CIs: Rao-Wu PSU bootstrap within
2021-2023 strata x mask resampling. Output: results/v3_external.json"""
import os, json, warnings
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from config import RESULTS, SEED, FEATS, EXT_CYCLES
from common import (load_cohort, load_all, empirical_pools, SCENARIOS, make_mask, apply_mask, fit_imputers, round_binary,
                    thresholds, wauc, raowu_weights)
warnings.filterwarnings("ignore")
K = int(os.environ.get("N_MASKS", 40)); B = int(os.environ.get("N_BOOT", 1000))
dev, ext = load_cohort(), load_cohort(cycles=EXT_CYCLES, weight="WTPH2YR")
_, flow_ext = load_cohort(return_flow=True, cycles=EXT_CYCLES, weight="WTPH2YR")
pools = empirical_pools(load_all())
X, y, w = dev[FEATS].to_numpy(float), dev.y.to_numpy(int), dev.w.to_numpy(float)
tr, ca = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED + 77).split(X, y, dev.strata))
pos = np.average(y[tr], weights=w[tr]); spw = (1 - pos) / pos; sw = w[tr] / w[tr].mean(); swb = sw * np.where(y[tr] == 1, spw, 1)
xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    reg_lambda=1.0, scale_pos_weight=spw, random_state=SEED, n_jobs=4, eval_metric="logloss", missing=np.nan).fit(X[tr], y[tr], sample_weight=sw)
sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr], sample_weight=swb)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=4).fit(X[tr], y[tr], sample_weight=swb)
PR = {"xgb": lambda Z: xgb.predict_proba(Z)[:, 1], "lr": lambda Z: lr.predict_proba(sc.transform(Z))[:, 1], "rf": lambda Z: rf.predict_proba(Z)[:, 1]}
THR = {m: thresholds(y[ca], f(X[ca]), w[ca]) for m, f in PR.items()}
imps = fit_imputers(X[tr], SEED, FEATS)
Xt, yt, wt, gt = ext[FEATS].to_numpy(float), ext.y.to_numpy(int), ext.w.to_numpy(float), ext.pir_tier.to_numpy(); nt = len(yt)
GR = ["low", "mid", "high", "all"]
POS = {t: ((gt == t) & (yt == 1)).astype(float) for t in GR[:3]}; POS["all"] = (yt == 1).astype(float)
NEG = {t: ((gt == t) & (yt == 0)).astype(float) for t in GR[:3]}; NEG["all"] = (yt == 0).astype(float)
Wr = raowu_weights(ext.strata.to_numpy(), ext.psu.to_numpy(), B, SEED + 3) * wt[None, :]
rng = np.random.default_rng(SEED + 4); A = np.zeros((B, K)); np.add.at(A, (np.repeat(np.arange(B), K), rng.integers(0, K, (B, K)).ravel()), 1); A /= K
def rate(v, vr, sel):
    return float((v * wt * sel).sum() / (wt * sel).sum()), ((Wr * sel) * vr).sum(1) / (Wr * sel).sum(1)
ci = lambda e, r: dict(est=e, lo=float(np.percentile(r, 2.5)), hi=float(np.percentile(r, 97.5)))
P0 = {m: f(Xt) for m, f in PR.items()}
out = {"flow_2021_2023": flow_ext, "n_test": nt, "n_strata": int(ext.strata.nunique()), "n_psu": int(ext.psu.nunique()),
       "ckd_cases": {t: int(POS[t].sum()) for t in GR}, "n_by_income": {t: int((gt == t).sum()) for t in GR[:3]},
       "thresholds_dev_calibration": THR, "auroc": {m: {t: wauc(yt[(gt == t) if t != "all" else slice(None)], P0[m][(gt == t) if t != "all" else slice(None)],
                                                       wt[(gt == t) if t != "all" else slice(None)]) for t in GR} for m in PR},
       "complete": {}, "scenarios": {}}
for m in PR:
    for th, v in THR[m].items():
        mc = (P0[m] < v).astype(float)
        out["complete"][f"{m}|{th}"] = {f"FNR_{t}": ci(*rate(mc, mc[None, :], POS[t])) for t in GR} | {f"FPR_{t}": ci(*rate(1 - mc, (1 - mc)[None, :], NEG[t])) for t in GR}
for sname in ["S1_cell", "S2_block", "S3_empirical", "S5_cell_unif", "S6_block_unif"]:
    MISS = {}
    for k in range(K):
        Xm = apply_mask(Xt, make_mask(SCENARIOS[sname], gt, SEED + 90000 + 101 * list(SCENARIOS).index(sname) + k, pools))
        Z = {"mean": imps["mean"].transform(Xm), "median": imps["median"].transform(Xm), "knn": round_binary(imps["knn"].transform(Xm))}
        mis = [round_binary(im.transform(Xm)) for im in imps["mi"]]
        for m, f in PR.items():
            ps = {meth: f(Z[meth]) for meth in Z}; ps["mice"] = np.mean([f(Zj) for Zj in mis], axis=0)
            if m == "xgb": ps["xgb_native"] = f(Xm)
            for meth, p in ps.items():
                for th, v in THR[m].items(): MISS.setdefault((m, meth, th), np.zeros((K, nt), np.float32))[k] = p < v
    for (m, meth, th), M in MISS.items():
        base = (P0[m] < THR[m][th]).astype(float); mo = M.mean(0); Mb = A @ M; res = {}; reps = {}
        for t in GR:
            fe, fr = rate(mo, Mb, POS[t]); be, br = rate(base, base[None, :], POS[t]); reps[t] = fr - br
            res[f"dFNR_{t}"] = ci(fe - be, fr - br)
            pe, prr = rate(1 - mo, 1 - Mb, NEG[t]); qe, qr = rate(1 - base, (1 - base)[None, :], NEG[t]); res[f"dFPR_{t}"] = ci(pe - qe, prr - qr)
        res["gap"] = ci(res["dFNR_low"]["est"] - res["dFNR_high"]["est"], reps["low"] - reps["high"])
        out["scenarios"][f"{sname}|{m}|{meth}|{th}"] = res
    print(sname, "done", flush=True)
json.dump(out, open(RESULTS / "v3_external.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ["flow_2021_2023", "n_test", "n_strata", "n_psu", "ckd_cases", "auroc"]}, indent=1, default=float))
for k in ["xgb|sens80"]: print(k, {a: round(b["est"], 3) for a, b in out["complete"][k].items()})
for k, v in out["scenarios"].items():
    if k.endswith("sens80") and k.split("|")[1] == "xgb": print(k, {a: (round(b["est"]*100, 1), round(b["lo"]*100, 1), round(b["hi"]*100, 1)) for a, b in v.items() if a in ("dFNR_low", "dFNR_high", "gap")})
