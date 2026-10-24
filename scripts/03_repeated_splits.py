#!/usr/bin/env python3
"""v3 main analysis: the whole pipeline repeated over R stratum-preserving PSU splits.
Per split: train / calibration / test sets are formed from whole survey strata (both PSUs of a stratum stay together, so
the within-stratum PSU structure is kept in the test set). Models + imputers are fit on train; thresholds on calibration;
K masks per missingness scenario are applied to the test set. Inference per split: Rao-Wu PSU bootstrap within test strata
x resampling of masks; pooled over splits by mixing split-specific bootstrap distributions (adds split-to-split variability).
Outputs results/v3_splits_raw.pkl (per-split summaries) — aggregated by 05_aggregate.py.
Env: N_SPLITS (20), N_MASKS (20), N_BOOT (100 per split), N_JOBS (5), RUN ('main' | 'ablation')."""
import os, sys, time, pickle, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, DMatrix
from config import RESULTS, SEED, FEATS, PROXIES, BLOCKS
from common import (load_cohort, load_all, empirical_pools, SCENARIOS, make_mask, apply_mask, fit_imputers, round_binary,
                    thresholds, wauc, raowu_weights, DELETABLE)
warnings.filterwarnings("ignore")
R_ = int(os.environ.get("N_SPLITS", 20)); K = int(os.environ.get("N_MASKS", 20)); B = int(os.environ.get("N_BOOT", 100))
NJ = int(os.environ.get("N_JOBS", 5)); RUN = os.environ.get("RUN", "main")
GR = ["low", "mid", "high", "all"]; THR = ["sens80", "youden", "fpr20"]
if RUN == "main":
    FE = FEATS; SCEN = list(SCENARIOS); MODELS_SCEN = {"S1_cell", "S2_block", "S3_empirical"}; AWARE = {"S1_cell", "S2_block", "S3_empirical"}
    MECH = True
else:  # proxy ablation: drop uric acid and albumin
    FE = [f for f in FEATS if f not in PROXIES]; SCEN = ["S1_cell", "S2_block", "S3_empirical"]; MODELS_SCEN = set(); AWARE = set(); MECH = False
FIX = [FEATS.index(f) for f in FE]
DEL_IDX = [i for i, f in enumerate(FE) if f in DELETABLE]
PT = np.round(np.arange(0.02, 0.61, 0.02), 2)

def xgb(spw, seed):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, n_jobs=1,
                         eval_metric="logloss", missing=np.nan)

def run_split(r, df, pools):
    t0 = time.time(); seed = SEED + 1000 * r
    X_full = df[FEATS].to_numpy(float); X = X_full[:, FIX]; y = df.y.to_numpy(int); w = df.w.to_numpy(float)
    g = df.pir_tier.to_numpy(); strata = df.strata.to_numpy(); psu = df.psu.to_numpy()
    trval, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, strata))
    a, b = next(GroupShuffleSplit(1, test_size=1 / 3, random_state=seed + 1).split(X[trval], y[trval], strata[trval]))
    tr, ca = trval[a], trval[b]
    assert not (set(strata[tr]) & set(strata[te]) or set(strata[ca]) & set(strata[te]) or set(strata[tr]) & set(strata[ca]))
    pos = np.average(y[tr], weights=w[tr]); spw = (1 - pos) / pos; sw = w[tr] / w[tr].mean()
    sw_bal = sw * np.where(y[tr] == 1, spw, 1.0)
    models = {"xgb": xgb(spw, seed).fit(X[tr], y[tr], sample_weight=sw)}
    if MODELS_SCEN:
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(max_iter=5000, C=1.0).fit(sc.transform(X[tr]), y[tr], sample_weight=sw_bal)
        models["lr"] = type("LRm", (), {"predict_proba": lambda self, Z, lr=lr, sc=sc: lr.predict_proba(sc.transform(Z))})()
        models["rf"] = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", random_state=seed,
                                              n_jobs=1).fit(X[tr], y[tr], sample_weight=sw_bal)
    pr = lambda m, Z: models[m].predict_proba(Z)[:, 1]
    THRS = {m: thresholds(y[ca], pr(m, X[ca]), w[ca]) for m in models}
    imps = fit_imputers(X[tr], seed, FE)
    Xt, yt, wt, gt = X[te], y[te], w[te], g[te]; nt = len(te); Xt_full = X_full[te]
    POS = {t: ((gt == t) & (yt == 1)).astype(float) for t in GR[:3]}; POS["all"] = (yt == 1).astype(float)
    NEG = {t: ((gt == t) & (yt == 0)).astype(float) for t in GR[:3]}; NEG["all"] = (yt == 0).astype(float)
    MEMB = {t: (gt == t) for t in GR[:3]}; MEMB["all"] = np.ones(nt, bool)
    # intersectional selections (exploratory)
    sx = np.where(df.female.to_numpy()[te] == 1, "female", "male"); ag = df.age3.to_numpy()[te]; rc = df.race5.to_numpy()[te]
    INTER = {}
    for t in ["low", "mid", "high"]:
        for lab, arr, levels in [("sex", sx, ["female", "male"]), ("age", ag, ["20-44", "45-64", "65+"]),
                                 ("race", rc, ["NHWhite", "NHBlack", "MexAm", "OtherHisp", "Other"])]:
            for lv in levels: INTER[f"{t}|{lab}|{lv}"] = ((gt == t) & (arr == lv) & (yt == 1)).astype(float)
    # complete-data reference
    P0 = {m: pr(m, Xt) for m in models}
    comp = {m: {k: (P0[m] < v).astype(np.float32) for k, v in THRS[m].items()} for m in models}
    # bootstrap machinery
    Wr = raowu_weights(strata[te], psu[te], B, seed + 7) * wt[None, :]
    rng = np.random.default_rng(seed + 8); A = np.zeros((B, K))
    np.add.at(A, (np.repeat(np.arange(B), K), rng.integers(0, K, (B, K)).ravel()), 1); A /= K
    def rate(vec_obs, vec_rep, sel):
        est = (vec_obs * wt * sel).sum() / (wt * sel).sum()
        reps = ((Wr * sel) * vec_rep).sum(1) / (Wr * sel).sum(1)
        return est, reps
    out = {"split": r, "n": dict(train=len(tr), cal=len(ca), test=nt), "n_strata": dict(train=len(set(strata[tr])),
           cal=len(set(strata[ca])), test=len(set(strata[te]))), "n_psu_test": len(set(psu[te])),
           "test_ckd": {t: int(POS[t].sum()) for t in GR}, "thresholds": THRS, "complete": {}, "keys": {}, "auroc": {},
           "dca": {}, "mech": {}, "single": {}, "inter": {}, "inter_n": {k: int(v.sum()) for k, v in INTER.items()}}
    for m in models:
        for th in THR:
            mc = comp[m][th]; d = {}
            for t in GR:
                d[f"FNR_{t}"] = rate(mc, mc[None, :], POS[t])
                d[f"FPR_{t}"] = rate(1 - mc, (1 - mc)[None, :], NEG[t])
            out["complete"][(m, th)] = d
        out["auroc"][("complete", m)] = {t: wauc(yt[MEMB[t]], P0[m][MEMB[t]], wt[MEMB[t]]) for t in GR}
    for k, sel in INTER.items():
        mc = comp["xgb"]["sens80"]; out["inter"][("complete", k)] = rate(mc, mc[None, :], sel)
    # NOTE: raw-score net-benefit curves (uncalibrated, class-weighted model) are computed here but NOT reported;
    # the reported decision curves use recalibrated probabilities (07_decision_curve.py).
    def nb_curve(p, sel_mask):
        yy, ww = yt[sel_mask], wt[sel_mask]; pp = p[sel_mask]; N = ww.sum()
        return np.array([((ww * ((pp >= c) & (yy == 1))).sum() - (ww * ((pp >= c) & (yy == 0))).sum() * c / (1 - c)) / N for c in PT])
    out["dca"][("complete", "xgb")] = {t: nb_curve(P0["xgb"], MEMB[t]) for t in ["low", "high", "all"]}
    prev = {t: np.average(yt[MEMB[t]], weights=wt[MEMB[t]]) for t in ["low", "high", "all"]}
    out["dca"][("treat_all",)] = {t: prev[t] - (1 - prev[t]) * PT / (1 - PT) for t in prev}
    # feature risk direction (training): sign of weighted correlation with CKD
    sign = np.array([np.sign(np.corrcoef(X[tr][:, j], y[tr])[0, 1]) for j in range(len(FE))])
    sd = X[tr].std(0); booster = models["xgb"].get_booster()
    shap = lambda Z: booster.predict(DMatrix(Z, missing=np.nan), pred_contribs=True)[:, :-1]
    S0 = shap(Xt) if MECH else None
    for sname in SCEN:
        spec = SCENARIOS[sname]; aware = {}
        if sname in AWARE:   # missingness-aware training: same scenario applied to train and calibration sets
            Dtr = make_mask(spec, g[tr], seed + 31, pools)[:, FIX]; Dca = make_mask(spec, g[ca], seed + 32, pools)[:, FIX]
            Xtr_m, Xca_m = apply_mask(X[tr], Dtr), apply_mask(X[ca], Dca)
            an = xgb(spw, seed).fit(Xtr_m, y[tr], sample_weight=sw)
            med = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr_m)
            aug = lambda Z, med=med: np.hstack([med.transform(Z), np.isnan(Z)[:, DEL_IDX].astype(float)])
            ai = xgb(spw, seed).fit(aug(Xtr_m), y[tr], sample_weight=sw)
            aware = {"xgb_aware_native": (lambda Z, an=an: an.predict_proba(Z)[:, 1], thresholds(y[ca], an.predict_proba(Xca_m)[:, 1], w[ca])),
                     "xgb_aware_ind": (lambda Z, ai=ai, aug=aug: ai.predict_proba(aug(Z))[:, 1], thresholds(y[ca], ai.predict_proba(aug(Xca_m))[:, 1], w[ca]))}
            out.setdefault("aware_thresholds", {})[sname] = {k: v[1] for k, v in aware.items()}
        mods = list(models) if sname in MODELS_SCEN else ["xgb"]
        MISS = {}; AUC = {}; DCA = {}; mech = {}
        for k in range(K):
            D = make_mask(spec, gt, seed + 5000 + 97 * list(SCENARIOS).index(sname) + k, pools)[:, FIX]
            Xm = apply_mask(Xt, D)
            Z = {"mean": imps["mean"].transform(Xm), "median": imps["median"].transform(Xm),
                 "knn": round_binary(imps["knn"].transform(Xm), FE)}
            mis = [round_binary(im.transform(Xm), FE) for im in imps["mi"]]
            probs = {}
            for m in mods:
                for meth in ["mean", "median", "knn"]: probs[(m, meth)] = (pr(m, Z[meth]), THRS[m])
                probs[(m, "mice")] = (np.mean([pr(m, Zj) for Zj in mis], axis=0), THRS[m])
            probs[("xgb", "xgb_native")] = (pr("xgb", Xm), THRS["xgb"])
            for an_, (f, th_) in aware.items(): probs[("xgb", an_)] = (f(Xm), th_)
            for key, (p, th_) in probs.items():
                for th in THR: MISS.setdefault(key + (th,), np.zeros((K, nt), np.float32))[k] = p < th_[th]
                if k < 5:
                    AUC.setdefault(key, []).append({t: wauc(yt[MEMB[t]], p[MEMB[t]], wt[MEMB[t]]) for t in GR})
                if key[0] == "xgb" and sname in ("S1_cell", "S2_block", "S3_empirical"):
                    DCA.setdefault(key, []).append({t: nb_curve(p, MEMB[t]) for t in ["low", "high", "all"]})
            if MECH and sname in ("S1_cell", "S2_block") and k < 10:
                flipped_base = (POS["low"] == 1) & (P0["xgb"] >= THRS["xgb"]["sens80"])
                for meth in ["mean", "median", "knn", "mice"]:
                    p = probs[("xgb", meth)][0]; flip = flipped_base & (p < THRS["xgb"]["sens80"])
                    stay = flipped_base & ~flip
                    Simp = np.mean([shap(Zj) for Zj in mis], axis=0) if meth == "mice" else shap(Z[meth])
                    Zv = np.mean(mis, axis=0) if meth == "mice" else Z[meth]
                    shift = (Zv - Xt) / sd * sign      # risk-signed standardized shift (negative = toward lower risk)
                    acc = mech.setdefault(meth, dict(n_flip=0, n_stay=0, dshap_flip=np.zeros(len(FE)), dshap_stay=np.zeros(len(FE)),
                                                     shift_flip=np.zeros(len(FE)), cnt_flip=np.zeros(len(FE)),
                                                     shift_stay=np.zeros(len(FE)), cnt_stay=np.zeros(len(FE))))
                    dS = Simp - S0
                    acc["n_flip"] += int(flip.sum()); acc["n_stay"] += int(stay.sum())
                    acc["dshap_flip"] += dS[flip].sum(0); acc["dshap_stay"] += dS[stay].sum(0)
                    acc["shift_flip"] += np.where(D[flip], shift[flip], 0).sum(0); acc["cnt_flip"] += D[flip].sum(0)
                    acc["shift_stay"] += np.where(D[stay], shift[stay], 0).sum(0); acc["cnt_stay"] += D[stay].sum(0)
        out["mech"][sname] = mech
        for key, arr in AUC.items(): out["auroc"][(sname,) + key] = {t: float(np.nanmean([a_[t] for a_ in arr])) for t in GR}
        for key, arr in DCA.items(): out["dca"][(sname,) + key] = {t: np.mean([a_[t] for a_ in arr], axis=0) for t in ["low", "high", "all"]}
        # summarize every key
        for key, M in MISS.items():
            m, meth, th = key; base = comp[m][th]; mo = M.mean(0); Mb = A @ M
            res = {}
            for t in GR:
                f_e, f_r = rate(mo, Mb, POS[t]); b_e, b_r = rate(base, base[None, :], POS[t])
                p_e, p_r = rate(1 - mo, 1 - Mb, NEG[t]); q_e, q_r = rate(1 - base, (1 - base)[None, :], NEG[t])
                res[f"FNR_{t}"] = (f_e, f_r.astype(np.float32)); res[f"dFNR_{t}"] = (f_e - b_e, (f_r - b_r).astype(np.float32))
                res[f"FPR_{t}"] = (p_e, p_r.astype(np.float32)); res[f"dFPR_{t}"] = (p_e - q_e, (p_r - q_r).astype(np.float32))
                res[f"flag_{t}"] = float(((1 - mo) * wt * MEMB[t]).sum() / (wt * MEMB[t]).sum())
            res["gap"] = (res["dFNR_low"][0] - res["dFNR_high"][0], res["dFNR_low"][1] - res["dFNR_high"][1])
            per = [((M[j] - base) * wt * POS["low"]).sum() / (wt * POS["low"]).sum() for j in range(K)]
            res["dFNR_low_per_mask"] = np.array(per, np.float32)
            out["keys"][(sname,) + key] = res
            if m == "xgb" and th == "sens80" and meth in ("mean", "median", "knn", "mice", "xgb_native") and sname in ("S1_cell", "S2_block"):
                for ik, sel in INTER.items():
                    if sel.sum() == 0: continue
                    f_e, f_r = rate(mo, Mb, sel); b_e, b_r = rate(base, base[None, :], sel)
                    out["inter"][(sname, meth, ik)] = (f_e - b_e, (f_r - b_r).astype(np.float32))
    # single-feature / single-block deletion for all low-income test participants (mechanism, xgb, sens80)
    if MECH:
        sets = {f: [FE.index(f)] for f in FE if f in DELETABLE}; sets.update({f"block:{b}": [FE.index(f) for f in fs] for b, fs in BLOCKS.items()})
        thr0 = THRS["xgb"]["sens80"]; base = comp["xgb"]["sens80"]; low = gt == "low"
        for nm, idx in sets.items():
            Xm = Xt.copy(); Xm[np.ix_(low, idx)] = np.nan
            cand = {"mean": pr("xgb", imps["mean"].transform(Xm)), "knn": pr("xgb", round_binary(imps["knn"].transform(Xm), FE)),
                    "mice": np.mean([pr("xgb", round_binary(im.transform(Xm), FE)) for im in imps["mi"]], axis=0)}
            for meth, p in cand.items():
                mm = (p < thr0).astype(float)
                out["single"][(nm, meth)] = float(((mm - base) * wt * POS["low"]).sum() / (wt * POS["low"]).sum())
    out["sec"] = time.time() - t0
    print(f"split {r} done in {out['sec']:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    df = load_cohort(); pools = empirical_pools(load_all())
    only = os.environ.get("ONLY_SPLITS")
    ids = [int(s) for s in only.split(",")] if only else list(range(R_))
    res = Parallel(n_jobs=NJ, verbose=0)(delayed(run_split)(r, df, pools) for r in ids)
    RESULTS.mkdir(exist_ok=True)
    fn = RESULTS / ("v3_splits_raw.pkl" if RUN == "main" else "v3_ablation_raw.pkl")
    pickle.dump(dict(meta=dict(n_splits=len(ids), n_masks=K, n_boot_per_split=B, run=RUN, feats=FE, pt=PT.tolist()), splits=res), open(fn, "wb"))
    print("saved", fn)
