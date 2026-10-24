#!/usr/bin/env python3
"""Decision-curve analysis (exploratory). Same 20 stratum-preserving splits and XGBoost model as 03_repeated_splits.py.
Because the models are trained with class weighting, raw scores are not calibrated probabilities; net benefit therefore
uses probabilities recalibrated on the calibration set (logistic recalibration of the logit, fit on complete calibration
data). Net benefit = TP/N - FP/N * pt/(1-pt), survey-weighted, on the test set, for complete data and each repair under
scenarios S1-S3 (10 masks each). Output: results/v3_dca.json (mean and 2.5/97.5 percentiles across splits)."""
import os, json, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from config import RESULTS, SEED, FEATS
from common import load_cohort, load_all, empirical_pools, SCENARIOS, make_mask, apply_mask, fit_imputers, round_binary
warnings.filterwarnings("ignore")
R_ = int(os.environ.get("N_SPLITS", 20)); K = int(os.environ.get("N_MASKS", 10)); NJ = int(os.environ.get("N_JOBS", 6))
PT = np.round(np.arange(0.05, 0.41, 0.01), 2)
def run(r, df, pools):
    seed = SEED + 1000 * r
    X = df[FEATS].to_numpy(float); y = df.y.to_numpy(int); w = df.w.to_numpy(float); g = df.pir_tier.to_numpy(); st = df.strata.to_numpy()
    trval, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, st))
    a, b = next(GroupShuffleSplit(1, test_size=1 / 3, random_state=seed + 1).split(X[trval], y[trval], st[trval])); tr, ca = trval[a], trval[b]
    pos = np.average(y[tr], weights=w[tr]); spw = (1 - pos) / pos
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                      reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, n_jobs=1, eval_metric="logloss", missing=np.nan)
    m.fit(X[tr], y[tr], sample_weight=w[tr] / w[tr].mean())
    lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    rc = LogisticRegression(C=1e6).fit(lg(m.predict_proba(X[ca])[:, 1])[:, None], y[ca], sample_weight=w[ca] / w[ca].mean())
    P = lambda Z: rc.predict_proba(lg(m.predict_proba(Z)[:, 1])[:, None])[:, 1]
    imps = fit_imputers(X[tr], seed)
    Xt, yt, wt, gt = X[te], y[te], w[te], g[te]
    sels = {"low": gt == "low", "high": gt == "high", "all": np.ones(len(te), bool)}
    def nb(p):
        return {t: [float(((wt[s] * ((p[s] >= c) & (yt[s] == 1))).sum() - (wt[s] * ((p[s] >= c) & (yt[s] == 0))).sum() * c / (1 - c)) / wt[s].sum()) for c in PT]
                for t, s in sels.items()}
    out = {"complete": nb(P(Xt))}
    prev = {t: np.average(yt[s], weights=wt[s]) for t, s in sels.items()}
    out["treat_all"] = {t: (prev[t] - (1 - prev[t]) * PT / (1 - PT)).tolist() for t in sels}
    for sname in ["S1_cell", "S2_block", "S3_empirical"]:
        acc = {}
        for k in range(K):
            Xm = apply_mask(Xt, make_mask(SCENARIOS[sname], gt, seed + 5000 + 97 * list(SCENARIOS).index(sname) + k, pools))
            ps = {"mean": P(imps["mean"].transform(Xm)), "median": P(imps["median"].transform(Xm)), "knn": P(round_binary(imps["knn"].transform(Xm))),
                  "mice": np.mean([P(round_binary(im.transform(Xm))) for im in imps["mi"]], axis=0), "xgb_native": P(Xm)}
            for meth, p in ps.items():
                for t, v in nb(p).items(): acc.setdefault(meth, {}).setdefault(t, []).append(v)
        for meth, d in acc.items(): out[f"{sname}|{meth}"] = {t: np.mean(v, axis=0).tolist() for t, v in d.items()}
    return out
if __name__ == "__main__":
    df = load_cohort(); pools = empirical_pools(load_all())
    res = Parallel(n_jobs=NJ)(delayed(run)(r, df, pools) for r in range(R_))
    agg = {"pt": PT.tolist(), "n_splits": R_, "n_masks": K, "curves": {}}
    for key in res[0]:
        agg["curves"][key] = {t: {"mean": np.mean([r[key][t] for r in res], 0).tolist(), "p2.5": np.percentile([r[key][t] for r in res], 2.5, 0).tolist(),
                                  "p97.5": np.percentile([r[key][t] for r in res], 97.5, 0).tolist()} for t in res[0][key]}
    # paired difference vs complete data at each pt (mean and split percentile range)
    agg["diff_vs_complete"] = {key: {t: {"mean": np.mean([np.array(r[key][t]) - np.array(r["complete"][t]) for r in res], 0).tolist(),
                                         "p2.5": np.percentile([np.array(r[key][t]) - np.array(r["complete"][t]) for r in res], 2.5, 0).tolist(),
                                         "p97.5": np.percentile([np.array(r[key][t]) - np.array(r["complete"][t]) for r in res], 97.5, 0).tolist()}
                               for t in res[0][key]} for key in res[0] if key not in ("complete", "treat_all")}
    json.dump(agg, open(RESULTS / "v3_dca.json", "w"), indent=1); print("saved v3_dca.json")
