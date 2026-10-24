"""Cohort construction, CKD label, missingness scenarios, imputers, thresholds and metric helpers (v3)."""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from config import HARMONIZED_CSV, CYCLES, FEATS, BINARY, PROTECTED, BLOCKS

def egfr_ckdepi2021(scr, age, female):
    kappa = np.where(female == 1, 0.7, 0.9); alpha = np.where(female == 1, -0.241, -0.302)
    return (142 * np.minimum(scr / kappa, 1) ** alpha * np.maximum(scr / kappa, 1) ** (-1.200)
            * 0.9938 ** age * np.where(female == 1, 1.012, 1.0))

def load_all(cycles=CYCLES):
    raw = pd.read_csv(HARMONIZED_CSV)
    df = raw[raw.cycle.isin(list(cycles))].copy()
    df["egfr"] = egfr_ckdepi2021(df.LBXSCR.to_numpy(float), df.age.to_numpy(float), df.female.to_numpy(float))
    df["ckd"] = ((df.egfr < 60) | (df.URDACT >= 30)).astype(float)
    df.loc[df.LBXSCR.isna() | df.URDACT.isna(), "ckd"] = np.nan
    df["pir_tier"] = df.pir_tier.astype("string")
    return df

def load_cohort(return_flow=False, cycles=CYCLES, weight="WTMEC2YR"):
    df = load_all(cycles)
    flow = {"participants": len(df), "MEC_examined": int((df.examined == 1).sum())}
    df = df[df.examined == 1]
    df = df[df.age >= 20]; flow["adults_20plus"] = len(df)
    df = df[df.pregnant != 1].copy(); flow["not_pregnant"] = len(df)
    flow["with_creatinine_and_uacr"] = int(df.ckd.notna().sum())
    flow["complete_15_features"] = int(df.dropna(subset=FEATS + ["ckd"]).shape[0])
    df = df.dropna(subset=FEATS + ["ckd", "pir_tier"]).reset_index(drop=True); flow["with_PIR"] = len(df)
    df["y"] = df.ckd.astype(int)
    wcol = weight if weight in df and df[weight].notna().all() else "WTMEC2YR"
    df["w"] = df[wcol] / len(cycles)
    df["strata"] = df.cycle.astype(str) + "_" + df.SDMVSTRA.astype(int).astype(str)
    df["psu"] = df.strata + "_" + df.SDMVPSU.astype(int).astype(str)
    df["pir_tier"] = df.pir_tier.astype(str)
    df["age3"] = pd.cut(df.age, [19, 44, 64, 200], labels=["20-44", "45-64", "65+"]).astype(str)
    return (df, flow) if return_flow else df

# ---------------- thresholds (chosen on the calibration set only) ----------------
def _grid(p): return np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
def thr_sens(y, p, w, target=0.80):
    best = None
    for t in _grid(p):
        s = np.average((p[y == 1] >= t), weights=w[y == 1])
        key = (0, s - target) if s + 1e-12 >= target else (1, target - s)
        if best is None or key < best[0]: best = (key, float(t))
    return best[1]
def thr_youden(y, p, w):
    best = None
    for t in _grid(p):
        j = np.average((p[y == 1] >= t), weights=w[y == 1]) + np.average((p[y == 0] < t), weights=w[y == 0]) - 1
        if best is None or j > best[0]: best = (j, float(t))
    return best[1]
def thr_fpr(y, p, w, target=0.20):
    best = None
    for t in _grid(p):
        f = np.average((p[y == 0] >= t), weights=w[y == 0])
        key = (0, target - f) if f <= target + 1e-12 else (1, f - target)
        if best is None or key < best[0]: best = (key, float(t))
    return best[1]
def thresholds(y, p, w): return {"sens80": thr_sens(y, p, w), "youden": thr_youden(y, p, w), "fpr20": thr_fpr(y, p, w)}

# ---------------- missingness scenarios ----------------
DELETABLE = [f for f in FEATS if f not in PROTECTED]
FI = {f: i for i, f in enumerate(FEATS)}
BIAS = {"low": 0.30, "mid": 0.10, "high": 0.05}; UNIF = {"low": 0.15, "mid": 0.15, "high": 0.15}
SCENARIOS = {
    "S1_cell":       dict(kind="cell", rates=BIAS, feats=DELETABLE),
    "S2_block":      dict(kind="block", rates=BIAS, blocks=list(BLOCKS), rho=0.5),
    "S3_empirical":  dict(kind="empirical", pooled=False),
    "S4_emp_pooled": dict(kind="empirical", pooled=True),
    "S5_cell_unif":  dict(kind="cell", rates=UNIF, feats=DELETABLE),
    "S6_block_unif": dict(kind="block", rates=UNIF, blocks=list(BLOCKS), rho=0.5),
    "S7_cell_all15": dict(kind="cell", rates=BIAS, feats=FEATS),
    "S8_lab_only":   dict(kind="block", rates=BIAS, blocks=["lab"], rho=0.5),
    "S9_exam_only":  dict(kind="block", rates=BIAS, blocks=["exam"], rho=0.5),
    "S10_quest_only": dict(kind="block", rates=BIAS, blocks=["questionnaire"], rho=0.5),
    # v4: preregistered H2 control exactly as preregistered (uniform 15%, all 15 features deletable); appended last so the
    # mask seeds of all earlier scenarios are unchanged
    "S11_cell_unif_all15": dict(kind="cell", rates=UNIF, feats=FEATS),
}

def empirical_pools(df_all):
    """Observed missingness patterns (13 deletable features) among ALL interviewed adults >=20 y, not pregnant, with PIR,
    2013-2018 (examined or not). Sampling weight = interview weight. Returns {tier or 'pooled': (patterns, probs)}."""
    d = df_all[(df_all.age >= 20) & (df_all.pregnant != 1) & df_all.pir_tier.notna()]
    M = d[DELETABLE].isna().to_numpy(); w = d.WTINT2YR.to_numpy(float); t = d.pir_tier.astype(str).to_numpy()
    pools = {}
    for g in ["low", "mid", "high", "pooled"]:
        s = np.ones(len(d), bool) if g == "pooled" else (t == g)
        pools[g] = (M[s], w[s] / w[s].sum())
    return pools

def make_mask(spec, tiers, seed, pools=None):
    """Boolean deletion matrix (n x len(FEATS)); common random numbers via the seed."""
    n = len(tiers); rng = np.random.default_rng(seed); D = np.zeros((n, len(FEATS)), bool)
    if spec["kind"] == "cell":
        r = np.array([spec["rates"][t] for t in tiers]); U = rng.random((n, len(spec["feats"])))
        for j, f in enumerate(spec["feats"]): D[:, FI[f]] = U[:, j] < r
    elif spec["kind"] == "block":
        r = np.array([spec["rates"][t] for t in tiers]); rho = spec["rho"]
        z = np.sqrt(rho) * rng.standard_normal((n, 1)) + np.sqrt(1 - rho) * rng.standard_normal((n, len(BLOCKS)))
        U = norm.cdf(z)
        for j, b in enumerate(BLOCKS):
            if b in spec["blocks"]:
                hit = U[:, j] < r
                for f in BLOCKS[b]: D[:, FI[f]] |= hit
    else:
        u = rng.random(n)
        for g in ["low", "mid", "high"]:
            s = tiers == g
            pat, pr = pools["pooled" if spec["pooled"] else g]
            idx = np.minimum(np.searchsorted(np.cumsum(pr), u[s]), len(pr) - 1)
            for j, f in enumerate(DELETABLE): D[s, FI[f]] = pat[idx, j]
    return D

def apply_mask(X, D):
    Xm = np.array(X, float, copy=True); Xm[D] = np.nan; return Xm

# ---------------- imputers ----------------
BIN_IDX = [FEATS.index(b) for b in BINARY]
def _bin_idx(feats): return [i for i, f in enumerate(feats) if f in BINARY]
class TypedSimpleImputer:
    """Mean or median for continuous features, most-frequent (mode) for binary features."""
    def __init__(self, strategy, feats=FEATS):
        self.b = _bin_idx(feats); self.c = [i for i in range(len(feats)) if i not in self.b]
        self.cont = SimpleImputer(strategy=strategy, keep_empty_features=True)
        self.bin = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
    def fit(self, X):
        self.cont.fit(X[:, self.c]); self.bin.fit(X[:, self.b]); return self
    def transform(self, X):
        out = np.array(X, float, copy=True)
        out[:, self.c] = self.cont.transform(X[:, self.c]); out[:, self.b] = self.bin.transform(X[:, self.b]); return out
def round_binary(X, feats=FEATS):
    out = np.array(X, float, copy=True); b = _bin_idx(feats)
    out[:, b] = np.clip(np.rint(out[:, b]), 0, 1); return out
def fit_imputers(X_train, seed, feats=FEATS, m=10):
    return {"mean": TypedSimpleImputer("mean", feats).fit(X_train), "median": TypedSimpleImputer("median", feats).fit(X_train),
            "knn": KNNImputer(n_neighbors=5).fit(X_train),
            "mi": [IterativeImputer(estimator=BayesianRidge(), max_iter=10, sample_posterior=True,
                                    random_state=seed + 100 + j).fit(X_train) for j in range(m)]}

def wauc(y, p, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) == 2 else np.nan

def wilson(k, n, z=1.959964):
    if n == 0: return (np.nan, np.nan)
    ph = k / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)

def raowu_weights(strata, psu, B, seed):
    """Rao-Wu rescaling bootstrap (m_h = n_h - 1): replicate weight multipliers, B x n."""
    rng = np.random.default_rng(seed); n = len(strata); R = np.zeros((B, n))
    groups = {}
    for i, (s, p) in enumerate(zip(strata, psu)): groups.setdefault(s, {}).setdefault(p, []).append(i)
    for s, ps in groups.items():
        lists = [np.array(v) for v in ps.values()]; nh = len(lists)
        if nh == 1:
            R[:, lists[0]] = 1.0; continue
        mh = nh - 1
        for b in range(B):
            for j in rng.integers(0, nh, mh): R[b, lists[j]] += nh / mh
    return R
