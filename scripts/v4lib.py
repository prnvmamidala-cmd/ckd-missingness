"""v4 shared pipeline: one stratum-preserving split -> fit models and imputers -> thresholds -> K masks per scenario ->
survey-weighted change in error rates. Used both for the full-sample point estimate (20 splits x 10 masks) and inside
every outer replicate of the hierarchical PSU bootstrap (11_v4_core.py)."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from xgboost import XGBClassifier
from config import FEATS, BINARY
from common import SCENARIOS, make_mask, apply_mask, TypedSimpleImputer, round_binary, thresholds, thr_sens

BIN = [FEATS.index(b) for b in BINARY]
GR = ["low", "mid", "high", "all"]
STATS = ["dFNR_low", "dFNR_mid", "dFNR_high", "dFNR_all", "dFPR_low", "dFPR_high", "gap"]

def xgb(spw, seed):
    return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=1.0, scale_pos_weight=spw, random_state=seed, n_jobs=1, eval_metric="logloss", missing=np.nan)

class LogisticChainedImputer:
    """Categorical-appropriate chained-equation imputer (v4 sensitivity, point D): every feature is regressed on all others
    in the (complete) training data -- logistic regression for binary features, Bayesian ridge for continuous ones. At
    transform time, missing cells start at the training mean/mode and are redrawn for n_iter rounds: binary cells as
    Bernoulli(p_hat) draws (always exactly 0 or 1, no rounding needed), continuous cells from the Bayesian ridge posterior
    predictive normal distribution."""
    def __init__(self, seed, n_iter=10): self.seed, self.n_iter = seed, n_iter
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9; Z = (X - self.mu) / self.sd; p = X.shape[1]
        self.init = X.mean(0).copy()
        for j in BIN: self.init[j] = float(np.round(X[:, j].mean()))
        self.models = []
        for j in range(p):
            o = [k for k in range(p) if k != j]
            m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[:, o], X[:, j].astype(int)) if j in BIN else BayesianRidge().fit(Z[:, o], Z[:, j])
            self.models.append((o, m))
        return self
    def transform(self, X):
        rng = np.random.default_rng(self.seed); M = np.isnan(X); Xc = np.where(M, self.init[None, :], X)
        cols = [j for j in range(X.shape[1]) if M[:, j].any()]
        for _ in range(self.n_iter):
            for j in cols:
                r = M[:, j]; o, m = self.models[j]; Zo = ((Xc[r][:, o] - self.mu[o]) / self.sd[o])
                if j in BIN: Xc[r, j] = (rng.random(r.sum()) < m.predict_proba(Zo)[:, 1]).astype(float)
                else:
                    mu, sd = m.predict(Zo, return_std=True); Xc[r, j] = (rng.normal(mu, sd)) * self.sd[j] + self.mu[j]
        return Xc

def fit_imputers(Xtr, seed, m=10, stochastic=True):
    out = {"mean": TypedSimpleImputer("mean").fit(Xtr), "median": TypedSimpleImputer("median").fit(Xtr),
           "knn": KNNImputer(n_neighbors=5).fit(Xtr),
           "det": IterativeImputer(estimator=BayesianRidge(), max_iter=10, sample_posterior=False, random_state=seed + 99).fit(Xtr),
           "sii": [], "lci": []}
    if stochastic:
        out["sii"] = [IterativeImputer(estimator=BayesianRidge(), max_iter=10, sample_posterior=True, random_state=seed + 100 + j).fit(Xtr) for j in range(m)]
        out["lci"] = [LogisticChainedImputer(seed + 300 + j).fit(Xtr) for j in range(m)]
    return out

def run_split(df, seed, K, scen=("S1_cell", "S2_block", "S3_empirical", "S7_cell_all15"), pools=None, other_models=("S1_cell", "S2_block"),
              keep_masks=False, idx=None, light=False, stochastic=True):
    """light=True (bootstrap replicates): XGBoost only, no fractional-binary variant; other_models ignored."""
    if light: other_models = ()
    X = df[FEATS].to_numpy(float); y = df.y.to_numpy(int); w = df.w.to_numpy(float); wi = df.w_ipw.to_numpy(float)
    g = df.pir_tier.to_numpy(); st = df.strata.to_numpy()
    if idx is None:
        trval, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=seed).split(X, y, st))
    else:
        trval, te = idx
    a, b = next(GroupShuffleSplit(1, test_size=1 / 3, random_state=seed + 1).split(X[trval], y[trval], st[trval])); tr, ca = trval[a], trval[b]
    pos = np.average(y[tr], weights=w[tr]); spw = (1 - pos) / pos; sw = w[tr] / w[tr].mean(); swb = sw * np.where(y[tr] == 1, spw, 1.0)
    models = {"xgb": xgb(spw, seed).fit(X[tr], y[tr], sample_weight=sw)}
    sc = StandardScaler().fit(X[tr]); lr = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr], sample_weight=swb)
    models["lr"] = type("L", (), {"predict_proba": lambda self, Z: lr.predict_proba(sc.transform(Z))})()
    if light: models["rf"] = None
    else: models["rf"] = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", random_state=seed, n_jobs=1).fit(X[tr], y[tr], sample_weight=swb)
    pr = lambda m, Z: models[m].predict_proba(Z)[:, 1]
    THR = {("xgb", k, "w"): v for k, v in thresholds(y[ca], pr("xgb", X[ca]), w[ca]).items()}
    THR[("xgb", "sens80", "ipw")] = thr_sens(y[ca], pr("xgb", X[ca]), wi[ca])
    if not light:
        for m in ("lr", "rf"): THR[(m, "sens80", "w")] = thr_sens(y[ca], pr(m, X[ca]), w[ca])
    else: models = {"xgb": models["xgb"]}
    imps = fit_imputers(X[tr], seed, stochastic=stochastic)
    Xt, yt, gt = X[te], y[te], g[te]; W = {"w": w[te], "ipw": wi[te]}
    POS = {t: ((gt == t) & (yt == 1)) for t in GR[:3]}; POS["all"] = yt == 1
    NEG = {t: ((gt == t) & (yt == 0)) for t in ["low", "high"]}
    P0 = {m: pr(m, Xt) for m in models}
    def rates(p, thr, ws):
        miss = p < thr; ww = W[ws]
        r = {f"FNR_{t}": (ww * miss * POS[t]).sum() / (ww * POS[t]).sum() for t in GR}
        for t in ["low", "high"]: r[f"FPR_{t}"] = (ww * (~miss) * NEG[t]).sum() / (ww * NEG[t]).sum()
        return r
    base = {k: rates(P0[k[0]], v, k[2]) for k, v in THR.items()}
    out = {"n": dict(train=len(tr), cal=len(ca), test=len(te)), "ckd_test": {t: int(POS[t].sum()) for t in GR}, "complete": base, "est": {}}
    for sname in scen:
        acc = {}
        for k in range(K):
            D = make_mask(SCENARIOS[sname], gt, seed + 5000 + 97 * list(SCENARIOS).index(sname) + k, pools); Xm = apply_mask(Xt, D)
            Z = {"mean": imps["mean"].transform(Xm), "median": imps["median"].transform(Xm), "knn": round_binary(imps["knn"].transform(Xm)),
                 "det": round_binary(imps["det"].transform(Xm))}
            sraw = [im.transform(Xm) for im in imps["sii"]]
            Zs = {"sii": [round_binary(s) for s in sraw],
                  "sii_frac": [] if light else [np.where(np.isin(np.arange(Xm.shape[1]), BIN)[None, :], np.clip(s, 0, 1), s) for s in sraw],
                  "lci": [im.transform(Xm) for im in imps["lci"]]}
            P = {}
            for meth in Z: P[("xgb", meth)] = pr("xgb", Z[meth])
            for meth in Zs:
                if Zs[meth]: P[("xgb", meth)] = np.mean([pr("xgb", z) for z in Zs[meth]], 0)
            P[("xgb", "native")] = pr("xgb", Xm)
            if sname in other_models:
                for m in ("lr", "rf"):
                    for meth in ("mean", "knn"): P[(m, meth)] = pr(m, Z[meth])
                    P[(m, "sii")] = np.mean([pr(m, z) for z in Zs["sii"]], 0)
            for (m, meth), p in P.items():
                for (mm, th, ws), thr in THR.items():
                    if mm != m: continue
                    if (ws == "ipw" or th != "sens80") and meth not in ("mean", "median", "knn", "sii", "native") + (("det",) if os.environ.get("IPW_DET") == "1" else ()): continue
                    r = rates(p, thr, ws); b0 = base[(mm, th, ws)]
                    v = [r["FNR_low"] - b0["FNR_low"], r["FNR_mid"] - b0["FNR_mid"], r["FNR_high"] - b0["FNR_high"], r["FNR_all"] - b0["FNR_all"],
                         r["FPR_low"] - b0["FPR_low"], r["FPR_high"] - b0["FPR_high"]]
                    v.append(v[0] - v[2]); acc.setdefault((sname, m, meth, th, ws), []).append(v)
        for key, v in acc.items():
            arr = np.array(v); out["est"][key] = arr if keep_masks else arr.mean(0)
    return out

def run_crossfit(df, seed, K, n_folds=4, **kw):
    """One outer-bootstrap evaluation: strata are randomly partitioned into n_folds folds; each fold is the test set once
    (the remaining strata are split 2:1 into training and calibration), so every stratum is tested exactly once. Fold-level
    estimates (each averaged over K masks) are averaged; complete-data baselines likewise."""
    st = df.strata.to_numpy(); S = np.unique(st); rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(S), np.arange(len(S)) % n_folds)); f = np.array([fold_of[s] for s in st])
    outs = [run_split(df, seed + 17 * (k + 1), K, idx=(np.flatnonzero(f != k), np.flatnonzero(f == k)), light=True, **kw) for k in range(n_folds)]
    est = {key: np.mean([o["est"][key] for o in outs], 0) for key in outs[0]["est"]}
    comp = {key: {s: float(np.mean([o["complete"][key][s] for o in outs])) for s in outs[0]["complete"][key]} for key in outs[0]["complete"]}
    return {"est": est, "complete": comp, "n_test": [o["n"]["test"] for o in outs], "ckd_test": [o["ckd_test"] for o in outs]}

def psu_resample(df, rng):
    """Outer bootstrap draw: within each stratum, draw n_h PSUs with replacement (n_h = 2 in every NHANES 2013-2018
    stratum) and stack copies; copies keep their stratum, so they always land on the same side of a stratum-grouped split."""
    parts = []
    for s, d in df.groupby("strata", sort=True):
        ps = d.psu.unique(); pick = rng.choice(ps, size=len(ps), replace=True)
        for j, p in enumerate(pick):
            c = d[d.psu == p].copy(); c["psu"] = f"{p}#{j}"; parts.append(c)
    return pd.concat(parts, ignore_index=True)
